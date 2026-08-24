"""Mini-batch BPR training with validation-driven early stopping.

This module exists to close the worst hole in the v11 thesis. That version
trained **every** model for exactly 10 epochs, including LightGCN, whose
reference implementation trains for 1000. A baseline stopped at 10 epochs has
not converged, so beating it proves nothing (CLAUDE.md muc "Hai sai sot cua cau
hinh cu"). The fix is structural rather than a bigger number:

* ``max_epochs`` is a ceiling of 1000, shared by every model;
* the actual stopping point is decided by the **validation** metric;
* ``curves.csv`` records loss and validation metric per evaluation, so a reader
  can see the plateau instead of taking convergence on trust.

Leakage rule 7 is asserted before the first epoch: model selection reads the
validation split and never test. The monitored metric name is checked too, so a
config that asked to monitor a test metric fails immediately rather than
producing results that quietly selected on test.

Patience is counted in **evaluations**, not epochs. With ``eval_every: 5`` and
``patience: 20`` a run tolerates 100 epochs without improvement before stopping.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd
import torch

from src.guards.leakage import assert_model_selection_scope
from src.training.loss import LOSS_BY_NAME, l2_regularization
from src.training.sampler import NegativeSampler
from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)


class TrainableGraphModel(Protocol):
    """What :class:`Trainer` needs from a model.

    Declared structurally rather than as a base class so the trainer stays
    unaware of which of the three graph variants it is holding -- the same
    reason :class:`~src.models.base.Recommender` exists for evaluation.
    """

    embeddings: torch.nn.Embedding
    name: str

    def propagate(self) -> torch.Tensor: ...

    def refresh_embeddings(self) -> None: ...

    def triple_scores(
        self, z: torch.Tensor, users: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]: ...


@dataclass
class TrainingResult:
    """Everything the run artifact needs to prove how training went.

    Attributes:
        curves: One row per evaluation -- ``epoch``, ``loss``, the validation
            metric, and a note. Written verbatim to ``curves.csv``.
        best_epoch: Epoch whose parameters were restored at the end.
        best_value: Validation metric at ``best_epoch``.
        n_epochs: Epochs actually run.
        stopped_early: Whether patience ran out before ``max_epochs``.
        monitor: Metric name that drove selection.
        seconds: Wall-clock training time.
    """

    curves: pd.DataFrame
    best_epoch: int
    best_value: float | None
    n_epochs: int
    stopped_early: bool
    monitor: str
    seconds: float
    losses: list[float] = field(default_factory=list)

    def describe(self) -> dict[str, object]:
        """Serialisable summary recorded inside ``metrics.json``."""
        return {
            "n_epochs": self.n_epochs,
            "best_epoch": self.best_epoch,
            "best_valid_metric": self.best_value,
            "monitor": self.monitor,
            "stopped_early": self.stopped_early,
            "first_loss": self.losses[0] if self.losses else None,
            "last_loss": self.losses[-1] if self.losses else None,
            "seconds": round(self.seconds, 1),
        }


class Trainer:
    """Trains one graph model with BPR over its observed interaction edges.

    The positive set is the aggregated edge table ``W(u,i)`` -- every observed
    train interaction, ``view`` included. Restricting positives to target
    behaviours would leave the graph propagating over view edges that never
    appear in the loss, so the supervision and the structure would disagree
    about what an interaction is (docs/DECISIONS.md muc D27).
    """

    def __init__(
        self,
        cfg: Config,
        model: TrainableGraphModel,
        sampler: NegativeSampler,
        users: np.ndarray,
        items: np.ndarray,
        weights: np.ndarray,
        validate: Callable[[TrainableGraphModel], float | None] | None,
        rng: np.random.Generator,
    ) -> None:
        """
        Args:
            cfg: Resolved configuration.
            model: The graph model being fitted.
            sampler: Negative sampler enforcing leakage rule 6.
            users: Visitor matrix index per positive edge.
            items: Item matrix index per positive edge.
            weights: ``W(u,i)`` per positive edge; used only by ``weighted_bpr``.
            validate: Callback returning the monitored validation metric, or
                ``None`` to train without early stopping.
            rng: Seeded generator driving the epoch shuffle.

        Raises:
            LeakageError: If ``cfg.training.monitor`` names a test metric.
            ValueError: If the positive arrays disagree in length.
        """
        if not (len(users) == len(items) == len(weights)):
            raise ValueError(
                f"canh duong lech nhau: {len(users):,} user, {len(items):,} item, "
                f"{len(weights):,} trong so"
            )
        # Rule 7, asserted before a single gradient step: whatever happens next,
        # selection was declared to read valid only.
        assert_model_selection_scope(cfg.training.monitor, consulted_splits=["valid"])

        self.cfg = cfg
        self.model = model
        self.sampler = sampler
        self.rng = rng
        self.validate = validate
        self.device = model.embeddings.weight.device

        self.users = torch.from_numpy(users).to(self.device)
        self.items = torch.from_numpy(items).to(self.device)
        self.weights = torch.from_numpy(weights).to(self.device)
        self.users_np = users

        self.loss_name = cfg.training.loss
        self.loss_fn = LOSS_BY_NAME[self.loss_name]
        self.optimizer = torch.optim.Adam(
            model.embeddings.parameters(), lr=cfg.training.learning_rate
        )

    # ──────────────────────────────────────────────────────────────────

    def train(self) -> TrainingResult:
        """Run the training loop and restore the best-validating parameters."""
        cfg = self.cfg.training
        n_positives = len(self.users_np)
        n_batches = max(1, int(np.ceil(n_positives / cfg.batch_size)))

        if self.validate is None:
            log.warning(
                "khong co callback validate — chay du %d epoch, KHONG co early stopping",
                cfg.max_epochs,
            )

        rows: list[dict[str, object]] = []
        losses: list[float] = []
        best_value: float | None = None
        best_epoch = 0
        best_state = copy.deepcopy(self.model.embeddings.state_dict())
        evals_without_gain = 0
        stopped_early = False
        started = time.time()
        epoch = 0

        log.info(
            "bat dau train %s: %s canh duong, %d batch/epoch, loss=%s, lr=%g, reg=%g",
            self.model.name, f"{n_positives:,}", n_batches, self.loss_name,
            cfg.learning_rate, cfg.reg_weight,
        )

        for epoch in range(1, cfg.max_epochs + 1):
            epoch_loss = self._run_epoch(n_positives, n_batches)
            losses.append(epoch_loss)

            if epoch % cfg.eval_every and epoch != cfg.max_epochs:
                continue

            value = self._validate_now()
            improved = self._is_improvement(value, best_value)
            if improved:
                best_value, best_epoch = value, epoch
                best_state = copy.deepcopy(self.model.embeddings.state_dict())
                evals_without_gain = 0
            elif value is not None:
                evals_without_gain += 1

            rows.append(
                {
                    "epoch": epoch,
                    "loss": epoch_loss,
                    f"valid_{cfg.monitor}": value,
                    "note": "best" if improved else f"khong cai thien ({evals_without_gain}/{cfg.patience})",
                }
            )
            log.info(
                "epoch %4d | loss %.6f | valid %s = %s%s",
                epoch, epoch_loss, cfg.monitor,
                "None" if value is None else f"{value:.6f}",
                "  <- best" if improved else "",
            )

            if evals_without_gain >= cfg.patience:
                stopped_early = True
                log.info(
                    "dung som o epoch %d: %d lan danh gia lien tiep khong cai thien",
                    epoch, evals_without_gain,
                )
                break

        # Restore only if validation actually chose something. Loading the
        # epoch-0 snapshot when no evaluation ever produced a number would
        # silently discard the whole run and leave the random initialisation
        # behind -- with a loss curve that still looks like it converged.
        if best_value is not None:
            self.model.embeddings.load_state_dict(best_state)
            log.info(
                "khoi phuc tham so cua epoch %d (valid %s = %.6f)",
                best_epoch, cfg.monitor, best_value,
            )
        else:
            best_epoch = epoch
            log.warning(
                "khong co gia tri valid nao — giu tham so cua epoch cuoi (%d), "
                "KHONG co chon mo hinh theo validation",
                epoch,
            )
        self.model.refresh_embeddings()

        return TrainingResult(
            curves=pd.DataFrame(rows),
            best_epoch=best_epoch,
            best_value=best_value,
            n_epochs=epoch,
            stopped_early=stopped_early,
            monitor=cfg.monitor,
            seconds=time.time() - started,
            losses=losses,
        )

    # ──────────────────────────────────────────────────────────────────

    def _run_epoch(self, n_positives: int, n_batches: int) -> float:
        """One pass over every positive edge in shuffled order."""
        cfg = self.cfg.training
        order = self.rng.permutation(n_positives)
        total_loss = 0.0

        for batch_index in range(n_batches):
            chunk = order[batch_index * cfg.batch_size : (batch_index + 1) * cfg.batch_size]
            if len(chunk) == 0:
                continue
            # Rule 6 is asserted on the first batch of every epoch; see
            # src/training/sampler.py for why not on every batch.
            negatives = self.sampler.sample(self.users_np[chunk], verify=batch_index == 0)

            positions = torch.from_numpy(chunk).to(self.device)
            users = self.users[positions]
            items = self.items[positions]
            negative_nodes = torch.from_numpy(negatives.astype("int64")).to(self.device)

            z = self.model.propagate()
            pos_scores, neg_scores, layer0 = self.model.triple_scores(
                z, users, items, negative_nodes
            )

            if self.loss_name == "weighted_bpr":
                edge_weights = self.weights[positions].repeat_interleave(negatives.shape[1])
                loss = self.loss_fn(pos_scores, neg_scores, edge_weights)
            else:
                loss = self.loss_fn(pos_scores, neg_scores)
            loss = loss + cfg.reg_weight * l2_regularization(*layer0)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.detach()) * len(chunk)

        return total_loss / n_positives

    def _validate_now(self) -> float | None:
        """Score the validation split with the current parameters."""
        if self.validate is None:
            return None
        self.model.refresh_embeddings()
        return self.validate(self.model)

    def _is_improvement(self, value: float | None, best: float | None) -> bool:
        """Whether ``value`` beats ``best`` under ``monitor_mode``."""
        if value is None:
            return False
        if best is None:
            return True
        return value > best if self.cfg.training.monitor_mode == "max" else value < best
