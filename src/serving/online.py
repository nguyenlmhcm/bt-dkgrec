"""Online serving: fold a visitor's latest behaviour into a trained model (D36).

The model is trained offline (Colab, periodically). What changes between two
trainings is the visitor's **history**, and that is exactly what this module
reads at request time -- the same split production recommenders use: weights
learnt offline, user representation recomputed from fresh behaviour online.

Why the user side can be recomputed **exactly**
-----------------------------------------------
Propagation (3.25)-(3.26) is ``z = mean_k(L_k)`` with ``L_k = A_hat . L_{k-1}``.
A visitor node is adjacent to items only, so its row of every layer is

    L_k[u] = sum_i a_hat[u, i] * L_{k-1}[i],   a_hat[u, i] = W_ui / sqrt(d_u * d_i)

Given the item rows of ``L_0 .. L_{K-1}`` -- precomputed once per training by
:func:`build_bundle` -- a visitor's ``z_u`` needs only their own edge weights
``W_ui`` at time ``tau``, which formula (3.17) gives in microseconds. For a
visitor already in the train graph and evaluated at ``tau = T_train`` the result
equals full propagation to float precision; ``tests/test_online.py`` asserts it.

What is approximated
--------------------
The **item** layers are those of the last training. A new click changes
``d_i`` (corrected here) and, in a full re-propagation, would also shift item
embeddings by a second-order amount; that part waits for the next periodic
training. This is the design the thesis describes: near-real-time graph update,
periodic offline training.

Nothing here imports torch: serving runs wherever numpy runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.graph.normalize import symmetric_normalize, weighted_degree
from src.graph.schema import NodeSpace
from src.graph.weighting import event_age_days

BEHAVIORS = ("view", "addtocart", "transaction")


@dataclass(frozen=True)
class ServingBundle:
    """Everything request-time scoring needs, computed once per training.

    Attributes:
        item_ids: Raw RetailRocket item id per item index.
        item_layers: ``(K, n_item, dim)`` -- item rows of ``L_0 .. L_{K-1}``.
        item_z: ``(n_item, dim)`` -- final item embeddings, the scoring targets.
        item_degree: ``(n_item,)`` weighted degree of each item in ``A``.
        alpha: Behaviour weights, ordered as :data:`BEHAVIORS`.
        lambda_decay: Daily decay of formula (3.17).
        d_day: Milliseconds per day.
    """

    item_ids: np.ndarray
    item_layers: np.ndarray
    item_z: np.ndarray
    item_degree: np.ndarray
    alpha: np.ndarray
    lambda_decay: float
    d_day: int

    @property
    def num_layers(self) -> int:
        return int(self.item_layers.shape[0])

    def save(self, path: Path) -> None:
        np.savez(
            path, item_ids=self.item_ids, item_layers=self.item_layers,
            item_z=self.item_z, item_degree=self.item_degree, alpha=self.alpha,
            lambda_decay=self.lambda_decay, d_day=self.d_day,
        )

    @classmethod
    def load(cls, path: Path) -> "ServingBundle":
        with np.load(path) as f:
            return cls(
                item_ids=f["item_ids"], item_layers=f["item_layers"], item_z=f["item_z"],
                item_degree=f["item_degree"], alpha=f["alpha"],
                lambda_decay=float(f["lambda_decay"]), d_day=int(f["d_day"]),
            )


def build_bundle(
    e0: np.ndarray,
    adjacency: sp.spmatrix,
    node_space: NodeSpace,
    num_layers: int,
    item_ids: np.ndarray,
    alpha: np.ndarray,
    lambda_decay: float,
    d_day: int,
) -> ServingBundle:
    """Propagate once on CPU and keep the item rows of every layer.

    Args:
        e0: Learnt layer-0 table, ``(n_nodes, dim)``.
        adjacency: The weighted ``A`` the model was trained on.
    """
    if e0.shape[0] != node_space.total or adjacency.shape[0] != node_space.total:
        raise ValueError(
            f"e0 {e0.shape} / A {adjacency.shape} khong khop khong gian node {node_space.total:,}"
        )
    a_hat = symmetric_normalize(adjacency)
    items = node_space.item_slice()

    layer = e0.astype("float32")
    pooled = layer.copy()
    item_layers = []
    for _ in range(num_layers):
        item_layers.append(layer[items].copy())
        layer = np.asarray(a_hat @ layer, dtype="float32")
        pooled += layer
    z = pooled / (num_layers + 1)

    return ServingBundle(
        item_ids=np.asarray(item_ids),
        item_layers=np.stack(item_layers),
        item_z=z[items],
        item_degree=weighted_degree(adjacency)[items].astype("float64"),
        alpha=np.asarray(alpha, dtype="float64"),
        lambda_decay=float(lambda_decay),
        d_day=int(d_day),
    )


def edge_weights_at(events: pd.DataFrame, tau: int, bundle: ServingBundle) -> pd.Series:
    """Formulas (3.16)-(3.18) at time ``tau``: one aggregated ``W_ui`` per item.

    Args:
        events: Columns ``item_idx``, ``behavior``, ``timestamp`` (ms). Events
            after ``tau`` are ignored -- the graph at ``tau`` cannot see them.

    Returns:
        ``W_ui`` indexed by ``item_idx``.
    """
    visible = events[events["timestamp"] <= tau]
    if visible.empty:
        return pd.Series(dtype="float64")
    codes = pd.Categorical(visible["behavior"], categories=list(BEHAVIORS)).codes
    if (codes < 0).any():
        raise ValueError(f"hanh vi la: {sorted(set(visible['behavior']) - set(BEHAVIORS))}")
    age = event_age_days(visible["timestamp"].to_numpy(), tau, bundle.d_day)
    weight = bundle.alpha[codes] * np.exp(-bundle.lambda_decay * age)
    return pd.Series(weight, index=visible["item_idx"].to_numpy()).groupby(level=0).sum()


def fold_in(
    bundle: ServingBundle,
    weights: pd.Series,
    e0_user: np.ndarray | None = None,
    weights_in_graph: pd.Series | None = None,
) -> np.ndarray:
    """Final embedding ``z_u`` of one visitor from their edge weights.

    Args:
        weights: ``W_ui`` at serving time, indexed by item index.
        e0_user: The visitor's learnt layer-0 row; ``None`` for a visitor the
            model never saw, whose layer 0 is then zero.
        weights_in_graph: The visitor's ``W_ui`` already counted inside
            ``item_degree`` (their train edges). Subtracted so an item's degree
            reflects the current weight once, not twice.
    """
    dim = bundle.item_z.shape[1]
    z = np.zeros(dim) if e0_user is None else np.asarray(e0_user, dtype="float64").copy()
    if weights.empty:
        return (z / (bundle.num_layers + 1)).astype("float32")

    idx = weights.index.to_numpy(dtype="int64")
    w = weights.to_numpy(dtype="float64")
    d_item = bundle.item_degree[idx].copy()
    if weights_in_graph is not None:
        d_item -= weights_in_graph.reindex(weights.index, fill_value=0.0).to_numpy()
    d_item += w
    coef = w / np.sqrt(w.sum() * d_item)

    for k in range(bundle.num_layers):
        z += coef @ bundle.item_layers[k][idx].astype("float64")
    return (z / (bundle.num_layers + 1)).astype("float32")


def top_k(scores: np.ndarray, k: int, exclude: np.ndarray | None = None) -> np.ndarray:
    """Best ``k`` item indices: score descending, then item index ascending (D32)."""
    scores = scores.astype("float64").copy()
    if exclude is not None and len(exclude):
        scores[exclude] = -np.inf
    k = min(k, len(scores))
    threshold = np.partition(scores, -k)[-k]
    candidates = np.flatnonzero(scores >= threshold)
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order[:k]]


def recommend(
    bundle: ServingBundle,
    events: pd.DataFrame,
    tau: int,
    k: int = 20,
    e0_user: np.ndarray | None = None,
    weights_in_graph: pd.Series | None = None,
    exclude_seen: bool = True,
) -> pd.DataFrame:
    """Top-K for one visitor at time ``tau`` from their event history.

    Seen items are excluded by default, matching ``filter_seen`` in evaluation.
    """
    weights = edge_weights_at(events, tau, bundle)
    z_u = fold_in(bundle, weights, e0_user, weights_in_graph)
    scores = bundle.item_z @ z_u
    seen = weights.index.to_numpy(dtype="int64") if exclude_seen else None
    best = top_k(scores, k, seen)
    return pd.DataFrame({
        "rank": np.arange(1, len(best) + 1),
        "item_idx": best,
        "item_id": bundle.item_ids[best],
        "score": scores[best],
    })
