"""Propagation paths behind a suggestion, read from the matrix the model propagates on.

With three layers, information reaches a visitor from an item three hops away along
``visitor -> item -> middle -> item``. The product of the normalised edge weights on one such
walk is one term of ``A_hat ** 3``. It shows where information flows; it is not a decomposition
of the score, which also depends on the learnt embeddings and on the other layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.graph.schema import NodeSpace


@dataclass(frozen=True)
class PathGraph:
    """Normalised training graph and the node space that gives each row its type."""

    a_hat: sp.csr_matrix
    space: NodeSpace
    item_degree: np.ndarray

    def kind(self, node: int) -> str:
        s = self.space
        if node < s.n_visitor:
            return "customer"
        if node < s.n_visitor + s.n_item:
            return "item"
        return "category" if node < s.n_visitor + s.n_item + s.n_category else "property"


def visitor_coefficients(weights: pd.Series, in_graph: pd.Series | None, item_degree: np.ndarray) -> pd.Series:
    """First hop ``W / sqrt(d_u * d_i)`` at the current time, as fold-in computes it."""
    if weights.empty:
        return weights
    idx = weights.index.to_numpy(dtype="int64")
    w = weights.to_numpy(dtype="float64")
    degree = item_degree[idx].copy()
    if in_graph is not None:
        degree -= in_graph.reindex(weights.index, fill_value=0.0).to_numpy()
    return pd.Series(w / np.sqrt(w.sum() * (degree + w)), index=weights.index)


def strongest_paths(graph: PathGraph, coefficients: pd.Series, target_item: int, visitor_node: int | None,
                    per_kind: int = 2) -> list[dict]:
    """Strongest walks ``visitor -> touched item -> middle -> target item``, best per middle type."""
    if coefficients.empty:
        return []
    offset = graph.space.n_visitor
    touched = coefficients.index.to_numpy(dtype="int64")
    target = offset + int(target_item)
    last_hop = graph.a_hat[target].tocoo()
    if last_hop.nnz == 0:
        return []
    keep = last_hop.col != visitor_node if visitor_node is not None else np.ones(last_hop.nnz, dtype=bool)
    middles, last_weights = last_hop.col[keep], last_hop.data[keep]
    hop = graph.a_hat[offset + touched][:, middles].tocoo()
    strength = coefficients.to_numpy()[hop.row] * hop.data * last_weights[hop.col]
    found: dict[str, list[dict]] = {}
    for at in np.argsort(-strength):
        middle = int(middles[hop.col[at]])
        kind = graph.kind(middle)
        bucket = found.setdefault(kind, [])
        if len(bucket) < per_kind:
            bucket.append({"kind": kind, "strength": float(strength[at]),
                           "item_idx": int(touched[hop.row[at]]), "middle": middle})
    return sorted((p for bucket in found.values() for p in bucket), key=lambda p: -p["strength"])
