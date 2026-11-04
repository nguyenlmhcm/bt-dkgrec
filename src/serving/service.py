"""Dich vu du bao Top-K cua demo (D38), theo thiet ke muc 3.7.1.

Moi yeu cau doc lich su cua khach tu kho do thi roi tinh lai Top-K bang fold-in.
Module nay khong biet Neo4j: lich su duoc truyen vao, nen test duoc khong can server.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data.mapping import IdMapping
from src.graph.schema import NodeSpace
from src.serving.online import BEHAVIORS, ServingBundle, build_bundle, edge_weights_at, recommend
from src.utils.config import load_config

#: Khoang cach giua hai su kien mo phong, ms.
STEP_MS = 1000


def event_weight(bundle: ServingBundle, behavior: str, age_days: float = 0.0) -> float:
    """Formula (3.17) for one event."""
    alpha = bundle.alpha[BEHAVIORS.index(behavior)]
    return float(alpha * np.exp(-bundle.lambda_decay * age_days))


@dataclass
class DemoService:
    """Model state for serving: item side from training, user side from history."""

    bundle: ServingBundle
    e0_visitors: np.ndarray
    visitor_ids: np.ndarray
    t_train: int

    def __post_init__(self) -> None:
        self._visitor_row = {int(v): i for i, v in enumerate(self.visitor_ids)}
        self._item_row = {int(v): i for i, v in enumerate(self.bundle.item_ids)}

    def next_timestamp(self, history: pd.DataFrame) -> int:
        """A new event happens just after everything already recorded."""
        last = int(history["timestamp"].max()) if len(history) else self.t_train
        return max(last, self.t_train) + STEP_MS

    def recommend(self, visitor_id: int, history: pd.DataFrame, k: int = 20) -> pd.DataFrame:
        """Top-K for a visitor from their current history (``item_id, behavior, timestamp``)."""
        known = history[history["item_id"].isin(self._item_row)].copy()
        known["item_idx"] = known["item_id"].map(self._item_row).astype("int64")
        tau = max(self.t_train, int(known["timestamp"].max()) if len(known) else self.t_train)
        row = self._visitor_row.get(int(visitor_id))
        e0_user = None if row is None else self.e0_visitors[row]
        in_graph = None
        if row is not None:
            in_graph = edge_weights_at(known[known["timestamp"] <= self.t_train], self.t_train, self.bundle)
        return recommend(self.bundle, known, tau, k=k, e0_user=e0_user, weights_in_graph=in_graph)

    @classmethod
    def from_artifacts(cls, cohort: str, model: str, e0_path: Path) -> "DemoService":
        """Build the service from the training graph and a saved ``E_0``."""
        cfg = load_config(model=model, cohort=cohort)
        paths = cfg.paths.resolved()
        graph_dir = paths["processed"] / cohort / model
        stats = json.loads((graph_dir / "graph_stats.json").read_text(encoding="utf-8"))
        space = NodeSpace(stats["n_visitor"], stats["n_item"], stats["n_category"], stats["n_property_value"])
        mapping = IdMapping.load(paths["interim"] / cohort)
        split = json.loads((paths["interim"] / cohort / "split.json").read_text(encoding="utf-8"))
        e0 = np.load(e0_path)
        alpha = np.array([cfg.weighting.alpha[b] for b in BEHAVIORS], dtype="float64")
        bundle = build_bundle(
            e0, sp.load_npz(graph_dir / "adjacency.npz"), space, cfg.model.num_layers,
            item_ids=mapping.item_ids, alpha=alpha,
            lambda_decay=cfg.weighting.lambda_decay, d_day=cfg.weighting.d_day,
        )
        return cls(bundle, e0[: space.n_visitor].copy(), mapping.visitor_ids, int(split["t_train"]))
