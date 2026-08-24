"""BT-DKGRec-GCN -- the proposed model.

Scoring follows KG_DESIGN.md muc 6.3, formulas (3.24)-(3.27)::

    h^(0)   = E                                  # layer-0 embedding table
    h^(l+1) = A_hat . h^(l)                      # parameter-free propagation
    z       = mean(h^(0) .. h^(L))               # mean pooling over layers
    s(u,i)  = z_u . z_i                          # inner product

**Only ``E`` is learnable.** There is no weight matrix and no non-linearity
between layers, in the LightGCN sense [He et al., SIGIR 2020]. That is a
deliberate constraint, not a simplification: it means any measured gain over
``lightgcn`` cannot come from extra network capacity, because there is none --
it can only come from what is written into ``A_hat``, which is where this
thesis's contribution lives (``w = alpha_b * exp(-lambda * dt)``).

Where the three graph models differ
-----------------------------------
Nowhere in this file. ``bt_dkgrec``, ``static_kg_gcn`` and ``lightgcn`` share
every line of this class; they differ only in the adjacency matrix built for
them in Buoc 4, which in turn differs only in
:meth:`~src.graph.weighting.EdgeWeighting.edge_weight` and in whether side
information is present. The subclasses added in Buoc 7 and Buoc 8 therefore
carry a name and nothing else -- which is precisely the controlled comparison
CLAUDE.md demands (``git diff`` khac dung mot bien).

Cold visitors
-------------
``supports_cold_start`` is ``False``. A visitor absent from the train mapping has
no row in ``E`` and no position in the graph, so there is no embedding to
propagate. Reporting a number for them would mean inventing one; the evaluator
reports their segment as null instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

from src.graph.normalize import symmetric_normalize
from src.graph.schema import NodeSpace
from src.models.base import ModelContext, NotFittedError, Recommender
from src.training.sampler import NegativeSampler
from src.training.trainer import TrainingResult, Trainer
from src.utils.logging import get_logger

log = get_logger(__name__)

#: Standard deviation of the layer-0 initialisation, as in the LightGCN
#: reference implementation. Recorded in ``describe()`` because it is a
#: hyperparameter of the result even though it never appears in a table.
INIT_STD = 0.1


class BTDKGRec(Recommender):
    """Weighted GCN over the projected knowledge graph, trained with BPR."""

    name = "bt_dkgrec"
    supports_cold_start = False

    def __init__(self) -> None:
        self.embeddings: nn.Embedding | None = None
        self.node_space: NodeSpace | None = None
        self.device = torch.device("cpu")
        self.num_layers = 0
        self.embedding_dim = 0
        self._a_hat: torch.Tensor | None = None
        self._z: torch.Tensor | None = None
        self._validate: Callable[["BTDKGRec"], dict[str, float] | None] | None = None
        self._result: TrainingResult | None = None
        self._sampler: NegativeSampler | None = None
        self._graph_stats: dict[str, object] = {}

    # ──────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────

    def attach_validation(
        self, validate: Callable[["BTDKGRec"], dict[str, float] | None]
    ) -> None:
        """Supply the validation callback used for early stopping.

        Passed in from the training script rather than carried on
        :class:`~src.models.base.ModelContext`, because the context is defined
        as train-derived material only and a validation hook is not that. The
        callback must read the **valid** split; leakage rule 7 is asserted
        inside :class:`~src.training.trainer.Trainer`.

        Args:
            validate: Callable receiving this model and returning the whole
                warm metric block of the validation split, or ``None`` when it
                cannot be computed. The trainer selects on
                ``cfg.training.monitor`` and records the remaining metrics as
                evidence (see :mod:`src.training.curves`).
        """
        self._validate = validate

    def _load_graph(self, context: ModelContext) -> sp.csr_matrix:
        """Read the adjacency matrix built for this exact model and cohort.

        Raises:
            FileNotFoundError: If Buoc 4 has not been run for this variant.
            ValueError: If the stored graph was built for a different model or
                cohort, or disagrees with the mapping about entity counts.
        """
        if context.graph_dir is None:
            raise ValueError(f"{self.name} can graph_dir — mo hinh do thi khong the chay khong graph")
        graph_dir = Path(context.graph_dir)
        adjacency_path = graph_dir / "adjacency.npz"
        stats_path = graph_dir / "graph_stats.json"
        if not adjacency_path.exists():
            raise FileNotFoundError(
                f"chua co do thi tai {graph_dir}: chay "
                f"`make graph COHORT={context.cfg.cohort.name}` truoc"
            )

        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats["model"] != self.name or stats["cohort"] != context.cfg.cohort.name:
            raise ValueError(
                f"graph tai {graph_dir} duoc dung cho {stats['cohort']}/{stats['model']}, "
                f"khong phai {context.cfg.cohort.name}/{self.name}"
            )
        if stats["n_visitor"] != context.n_visitors or stats["n_item"] != context.n_items:
            raise ValueError(
                f"graph co {stats['n_visitor']:,} visitor / {stats['n_item']:,} item nhung "
                f"mapping co {context.n_visitors:,} / {context.n_items:,} — hai ben lech nhau"
            )

        self._graph_stats = stats
        self.node_space = NodeSpace(
            n_visitor=int(stats["n_visitor"]),
            n_item=int(stats["n_item"]),
            n_category=int(stats["n_category"]),
            n_property_value=int(stats["n_property_value"]),
        )
        adjacency = sp.load_npz(adjacency_path)
        if adjacency.shape[0] != self.node_space.total:
            raise ValueError(
                f"ma tran ke {adjacency.shape} khong khop khong gian node "
                f"({self.node_space.total:,})"
            )
        return adjacency.tocsr()

    def _prepare(self, context: ModelContext) -> None:
        """Build ``A_hat`` on the target device and initialise ``E``."""
        cfg = context.cfg
        self.device = torch.device(cfg.training.device)
        self.embedding_dim = int(cfg.model.embedding_dim)
        self.num_layers = int(cfg.model.num_layers)

        adjacency = self._load_graph(context)
        # A_hat is recomputed here rather than loaded: it is a deterministic
        # function of A, and a stale copy on disk could silently disagree with
        # the matrix it came from (src/graph/builder.py, save_graph).
        normalized = symmetric_normalize(adjacency).tocoo()
        indices = torch.from_numpy(
            np.vstack([normalized.row, normalized.col]).astype("int64")
        )
        values = torch.from_numpy(normalized.data.astype("float32"))
        # check_invariants is opted into rather than left implicit: it validates
        # the index arrays once at construction, and a malformed sparse tensor
        # fails as SEGFAULT during propagation rather than as an exception.
        self._a_hat = torch.sparse_coo_tensor(
            indices, values, size=normalized.shape, device=self.device,
            check_invariants=True,
        ).coalesce()

        assert self.node_space is not None
        self.embeddings = nn.Embedding(self.node_space.total, self.embedding_dim).to(self.device)
        nn.init.normal_(self.embeddings.weight, std=INIT_STD)
        log.info(
            "%s: %s node x dim %d, %d lop lan truyen, A_hat co %s phan tu khac 0, device=%s",
            self.name, f"{self.node_space.total:,}", self.embedding_dim, self.num_layers,
            f"{self._a_hat._nnz():,}", self.device,
        )

    # ──────────────────────────────────────────────────────────────────
    # Propagation and scoring
    # ──────────────────────────────────────────────────────────────────

    def propagate(self) -> torch.Tensor:
        """Formulas (3.25)-(3.26): propagate and mean-pool over layers.

        Returns:
            ``z`` of shape ``(n_nodes, dim)``. Differentiable with respect to
            the layer-0 table, which is the only parameter in the model.
        """
        if self.embeddings is None or self._a_hat is None:
            raise NotFittedError(f"{self.name} chua dung do thi — goi fit() truoc")
        layer = self.embeddings.weight
        pooled = layer
        for _ in range(self.num_layers):
            layer = torch.sparse.mm(self._a_hat, layer)
            pooled = pooled + layer
        return pooled / (self.num_layers + 1)

    def refresh_embeddings(self) -> None:
        """Recompute and cache ``z`` for scoring.

        Called by the trainer before every validation pass and once at the end
        of training. Without the refresh, :meth:`score` would rank with the
        embeddings of whichever epoch last cached them -- a stale-state bug that
        produces plausible but wrong numbers.
        """
        with torch.no_grad():
            self._z = self.propagate().detach()

    def triple_scores(
        self, z: torch.Tensor, users: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        """Score one batch of BPR triples.

        Args:
            z: Propagated embeddings from :meth:`propagate`.
            users: Visitor matrix indices, shape ``(n,)``.
            positives: Observed item matrix indices, shape ``(n,)``.
            negatives: Sampled item matrix indices, shape ``(n, num_negatives)``.

        Returns:
            ``(pos_scores, neg_scores, layer0_embeddings)``. The first two are
            flattened to ``(n * num_negatives,)`` so every negative is compared
            against its own positive. The third is the layer-0 embeddings of the
            batch, which is what the L2 penalty acts on -- regularising ``z``
            instead would penalise the graph structure rather than the
            parameters.
        """
        assert self.node_space is not None and self.embeddings is not None
        num_negatives = negatives.shape[1]
        item_offset = self.node_space.item_offset

        user_nodes = users
        pos_nodes = positives + item_offset
        neg_nodes = negatives.reshape(-1) + item_offset

        z_user = z[user_nodes]
        z_pos = z[pos_nodes]
        z_neg = z[neg_nodes]

        pos_scores = (z_user * z_pos).sum(dim=1)
        repeated_user = z_user.repeat_interleave(num_negatives, dim=0)
        neg_scores = (repeated_user * z_neg).sum(dim=1)
        if num_negatives > 1:
            pos_scores = pos_scores.repeat_interleave(num_negatives)

        table = self.embeddings.weight
        layer0 = (table[user_nodes], table[pos_nodes], table[neg_nodes])
        return pos_scores, neg_scores, layer0

    def score(self, visitor_indices: np.ndarray) -> np.ndarray:
        """Score every train item for a batch of visitors.

        Raises:
            NotFittedError: If called before :meth:`fit`.
            ValueError: If a cold visitor (``-1``) reaches a model that declared
                it cannot serve one.
        """
        if self._z is None or self.node_space is None:
            raise NotFittedError(f"{self.name} chua duoc fit")
        indices = np.asarray(visitor_indices)
        if (indices < 0).any():
            raise ValueError(
                f"{self.name} khong phuc vu visitor cold nhung nhan {int((indices < 0).sum())} "
                "chi so -1 — evaluator phai loc truoc"
            )
        with torch.no_grad():
            z_user = self._z[torch.from_numpy(indices.astype("int64")).to(self.device)]
            z_item = self._z[self.node_space.item_slice()]
            scores = z_user @ z_item.T
        return scores.cpu().numpy()

    # ──────────────────────────────────────────────────────────────────
    # Fitting
    # ──────────────────────────────────────────────────────────────────

    def fit(self, context: ModelContext) -> None:
        """Train on train data only, selecting the epoch by validation metric."""
        self._prepare(context)
        edges = context.interaction_edges
        if edges.empty:
            raise ValueError(
                f"{self.name}: khong co canh tuong tac nao — chay `make graph` truoc"
            )

        rng = np.random.default_rng(context.cfg.seed)
        seen = self._seen_matrix(edges, context.n_visitors, context.n_items)
        self._sampler = NegativeSampler(
            seen=seen,
            item_ids=context.mapping.item_ids,
            num_negatives=context.cfg.training.num_negatives,
            rng=rng,
        )

        trainer = Trainer(
            cfg=context.cfg,
            model=self,
            sampler=self._sampler,
            users=edges["visitor_idx"].to_numpy().astype("int64"),
            items=edges["item_idx"].to_numpy().astype("int64"),
            weights=edges["weight"].to_numpy().astype("float32"),
            validate=self._validate,
            rng=rng,
        )
        self._result = trainer.train()
        self.refresh_embeddings()

    @staticmethod
    def _seen_matrix(edges: pd.DataFrame, n_visitors: int, n_items: int) -> sp.csr_matrix:
        """Train interactions as a sparse mask, used to reject false negatives.

        Built from the aggregated edges rather than from the raw events: the
        aggregation is grouped by ``(visitor_idx, item_idx)``, so the two carry
        exactly the same support while the edge table is an order of magnitude
        smaller.
        """
        matrix = sp.csr_matrix(
            (
                np.ones(len(edges), dtype="bool"),
                (edges["visitor_idx"].to_numpy(), edges["item_idx"].to_numpy()),
            ),
            shape=(n_visitors, n_items),
        )
        matrix.sum_duplicates()
        return matrix

    @property
    def training_result(self) -> TrainingResult | None:
        """Curves and stopping facts of the last :meth:`fit`."""
        return self._result

    def describe(self) -> dict[str, object]:
        record: dict[str, object] = {
            **super().describe(),
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "init_std": INIT_STD,
            "learnable": "layer-0 embeddings only (khong co ma tran trong so giua cac lop)",
            "n_nodes": self.node_space.total if self.node_space else None,
            "weighting": self._graph_stats.get("weighting"),
        }
        if self._sampler is not None:
            record["negative_sampling"] = self._sampler.describe()
        if self._result is not None:
            record["training"] = self._result.describe()
        return record
