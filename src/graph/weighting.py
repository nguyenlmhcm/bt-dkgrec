"""Behavior-time edge weighting -- the single definition of the formula.

This module is the **only** place the weighting formula exists. Nothing else in
the codebase may re-derive it; builder, models, exporter and the demo all call
in here (CLAUDE.md, DRY rule 2).

Formulas, numbered as in the thesis::

    (3.16)  dt(u,i,b,t) = max(0, (T_train - t) / D_day)        # event age, days
    (3.17)  w(u,i,b,t)  = alpha_b * exp(-lambda * dt)          # event weight
    (3.18)  W(u,i)      = sum over (b,t) in E_train(u,i) of w  # aggregated edge

Academic lineage: the behavior weights alpha_b follow MBGCN [Jin et al., SIGIR
2020] and KHGT [Xia et al., AAAI 2021]; the time-decay term follows KHGT. The
contribution of this thesis is *where* the decay is applied -- KHGT encodes time
inside the network (temporal encoding plus graph attention), while here it is
encoded directly into the **edge weight of the knowledge graph**, which keeps
propagation parameter-free in the LightGCN sense [He et al., SIGIR 2020].

Ablation design
---------------
:class:`UniformWeighting` **inherits** :class:`BehaviorTimeWeighting` and
overrides exactly one method, ``edge_weight``. Nothing else changes -- not the
builder, not the aggregation, not the normalisation. That is what makes
``static_kg_gcn`` vs ``bt_dkgrec`` a controlled comparison with a single moving
part (KG_DESIGN.md muc 4.3, docs/DECISIONS.md muc D4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)


def event_age_days(timestamps: np.ndarray, t_train: int, d_day: int) -> np.ndarray:
    """Formula (3.16): age of an event at the train boundary, in days.

    Args:
        timestamps: Event timestamps in milliseconds.
        t_train: End of the training window, in milliseconds.
        d_day: Milliseconds per day (86,400,000 for RetailRocket).

    Returns:
        Non-negative ages. Clipping at zero means an event exactly on the
        boundary decays by nothing rather than gaining weight.
    """
    age = (float(t_train) - timestamps.astype("float64")) / float(d_day)
    return np.maximum(age, 0.0)


class EdgeWeighting(ABC):
    """Strategy computing the weight of one interaction event."""

    name: str = "abstract"

    @abstractmethod
    def edge_weight(self, behavior_codes: np.ndarray, delta_days: np.ndarray) -> np.ndarray:
        """Weight of each event.

        Args:
            behavior_codes: Integer code of the behavior, indexing ``alpha``.
            delta_days: Event age in days, from :func:`event_age_days`.

        Returns:
            One weight per event, strictly positive.
        """

    def describe(self) -> dict[str, object]:
        """Serialisable description recorded in ``graph_stats.json``."""
        return {"weighting": self.name}


class BehaviorTimeWeighting(EdgeWeighting):
    """Formula (3.17): ``w = alpha_b * exp(-lambda * dt)``.

    This is the proposed model's contribution. Parameters come from
    ``configs/base.yaml`` and are never hardcoded.
    """

    name = "behavior_time"

    def __init__(self, alpha: np.ndarray, lambda_decay: float, behaviors: tuple[str, ...]) -> None:
        """
        Args:
            alpha: Behavior weights ordered to match ``behaviors``.
            lambda_decay: Daily decay coefficient.
            behaviors: Behavior labels, position = behavior code.
        """
        if np.any(alpha <= 0):
            raise ValueError(f"behavior weights must be > 0, got {alpha}")
        if lambda_decay < 0:
            raise ValueError(f"lambda must be >= 0, got {lambda_decay}")
        self.alpha = alpha.astype("float64")
        self.lambda_decay = float(lambda_decay)
        self.behaviors = behaviors

    def edge_weight(self, behavior_codes: np.ndarray, delta_days: np.ndarray) -> np.ndarray:
        return self.alpha[behavior_codes] * np.exp(-self.lambda_decay * delta_days)

    def describe(self) -> dict[str, object]:
        return {
            "weighting": self.name,
            "alpha": dict(zip(self.behaviors, self.alpha.tolist(), strict=True)),
            "lambda_decay": self.lambda_decay,
        }


class UniformWeighting(BehaviorTimeWeighting):
    """Ablation: drop both the behavior signal and the time decay.

    Inherits every part of :class:`BehaviorTimeWeighting` and overrides **only**
    :meth:`edge_weight`. Used by ``static_kg_gcn`` (knowledge graph without
    behavior-time) and by ``lightgcn`` (plain collaborative filtering).
    """

    name = "uniform"

    def edge_weight(self, behavior_codes: np.ndarray, delta_days: np.ndarray) -> np.ndarray:
        return np.ones(len(behavior_codes), dtype="float64")

    def describe(self) -> dict[str, object]:
        return {"weighting": self.name, "note": "edge_weight() == 1.0 (ablation)"}


#: Which weighting each model uses. The behaviour lives in code, never in YAML,
#: so no configuration mistake can run ``bt_dkgrec`` with weighting disabled
#: (docs/DECISIONS.md muc D4). In Buoc 6/7 the model classes inherit from these.
WEIGHTING_BY_MODEL: dict[str, type[BehaviorTimeWeighting]] = {
    "bt_dkgrec": BehaviorTimeWeighting,
    "bt_dkgrec_l05": BehaviorTimeWeighting,
    "static_kg_gcn": UniformWeighting,
    "lightgcn": UniformWeighting,
}


def weighting_for_model(cfg: Config) -> BehaviorTimeWeighting:
    """Build the weighting strategy of the configured model.

    Raises:
        ValueError: If the model has no graph (the heuristic baselines).
    """
    model = cfg.model.name
    if model not in WEIGHTING_BY_MODEL:
        raise ValueError(
            f"model {model!r} khong dung do thi; chi {sorted(WEIGHTING_BY_MODEL)} co graph"
        )
    behaviors = tuple(cfg.data.history_behaviors)
    alpha = np.array([cfg.weighting.alpha[b] for b in behaviors], dtype="float64")
    strategy = WEIGHTING_BY_MODEL[model](
        alpha=alpha, lambda_decay=cfg.weighting.lambda_decay, behaviors=behaviors
    )
    log.info("weighting cua %s: %s", model, strategy.describe())
    return strategy


def aggregate_interaction_edges(
    train_events: pd.DataFrame,
    weighting: EdgeWeighting,
    t_train: int,
    d_day: int,
    behaviors: tuple[str, ...],
    target_behaviors: tuple[str, ...],
) -> pd.DataFrame:
    """Formula (3.18): collapse train events into weighted visitor-item edges.

    Every train event contributes, ``view`` included: views are the historical
    signal, while ``addtocart``/``transaction`` additionally define the ground
    truth used at evaluation time.

    Args:
        train_events: Train events with ``visitor_idx``, ``item_idx``,
            ``behavior`` and ``timestamp``.
        weighting: Strategy supplying formula (3.17).
        t_train: End of the training window.
        d_day: Milliseconds per day.
        behaviors: Behavior labels, position = behavior code.
        target_behaviors: Labels counted as target behaviors.

    Returns:
        One row per ``(visitor_idx, item_idx)`` with the aggregated ``weight``,
        per-behavior counts, and ``last_ts`` -- the properties Neo4j stores on
        ``INTERACTED_WITH`` (KG_DESIGN.md muc 3).
    """
    codes = pd.Categorical(train_events["behavior"], categories=list(behaviors)).codes
    if (codes < 0).any():
        unknown = set(train_events["behavior"]) - set(behaviors)
        raise ValueError(f"behavior khong nam trong cau hinh: {sorted(unknown)}")

    stamps = train_events["timestamp"].to_numpy()
    delta = event_age_days(stamps, t_train, d_day)
    weights = weighting.edge_weight(codes, delta)

    frame = pd.DataFrame(
        {
            "visitor_idx": train_events["visitor_idx"].to_numpy(),
            "item_idx": train_events["item_idx"].to_numpy(),
            "weight": weights,
            "n_view": (codes == behaviors.index("view")).astype("int32"),
            "n_cart": (codes == behaviors.index("addtocart")).astype("int32"),
            "n_txn": (codes == behaviors.index("transaction")).astype("int32"),
            "last_ts": stamps,
        }
    )
    edges = frame.groupby(["visitor_idx", "item_idx"], sort=False, observed=True).agg(
        weight=("weight", "sum"),
        n_view=("n_view", "sum"),
        n_cart=("n_cart", "sum"),
        n_txn=("n_txn", "sum"),
        last_ts=("last_ts", "max"),
    ).reset_index()

    n_target = int(train_events["behavior"].isin(target_behaviors).sum())
    log.info(
        "W(u,i): %s su kien -> %s canh (trong do %s su kien muc tieu), w in [%.6f, %.6f]",
        f"{len(train_events):,}", f"{len(edges):,}", f"{n_target:,}",
        edges["weight"].min(), edges["weight"].max(),
    )
    return edges
