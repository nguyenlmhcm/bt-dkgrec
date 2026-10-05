"""Online serving (D36): fold-in must agree with full propagation where it claims to.

The claim in :mod:`src.serving.online` is precise: for a visitor already in the
train graph, at ``tau = T_train``, folding their history in reproduces the
``z_u`` that full propagation gives. If that equality breaks, the demo would be
recommending from a different model than the one evaluated in Chapter 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.graph.normalize import symmetric_normalize
from src.graph.schema import NodeSpace
from src.serving.online import (
    ServingBundle,
    build_bundle,
    edge_weights_at,
    fold_in,
    recommend,
    top_k,
)

DAY = 86_400_000
T_TRAIN = 100 * DAY
ALPHA = np.array([1.0, 2.0, 3.0])
LAMBDA = 0.05
SPACE = NodeSpace(n_visitor=4, n_item=6, n_category=2, n_property_value=0)

#: visitor -> (item, behavior, age in days at T_TRAIN)
HISTORY = {
    0: [(0, "view", 1), (0, "view", 10), (2, "addtocart", 3)],
    1: [(1, "transaction", 0), (3, "view", 40)],
    2: [(4, "view", 2)],
    3: [(2, "view", 5), (5, "transaction", 20), (1, "view", 1)],
}
#: item -> category, the knowledge-graph side of the adjacency
CATEGORY = [0, 0, 1, 1, 0, 1]


def _events(visitor: int) -> pd.DataFrame:
    rows = HISTORY[visitor]
    return pd.DataFrame({
        "item_idx": [r[0] for r in rows],
        "behavior": [r[1] for r in rows],
        "timestamp": [T_TRAIN - r[2] * DAY for r in rows],
    })


def _skeleton() -> ServingBundle:
    """Bundle carrying only what edge_weights_at reads."""
    return ServingBundle(
        item_ids=np.arange(SPACE.n_item), item_layers=np.zeros((1, SPACE.n_item, 1)),
        item_z=np.zeros((SPACE.n_item, 1)), item_degree=np.ones(SPACE.n_item),
        alpha=ALPHA, lambda_decay=LAMBDA, d_day=DAY,
    )


def _graph() -> tuple[sp.csr_matrix, dict[int, pd.Series]]:
    """Weighted A built from HISTORY with the serving formula at T_TRAIN."""
    n = SPACE.total
    rows, cols, vals = [], [], []
    weights = {}
    for visitor in HISTORY:
        w = edge_weights_at(_events(visitor), T_TRAIN, _skeleton())
        weights[visitor] = w
        for item, value in w.items():
            node = SPACE.item_offset + int(item)
            rows += [visitor, node]
            cols += [node, visitor]
            vals += [value, value]
    for item, category in enumerate(CATEGORY):
        a, b = SPACE.item_offset + item, SPACE.category_offset + category
        rows += [a, b]
        cols += [b, a]
        vals += [1.0, 1.0]
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n)), weights


def _full_propagation(e0: np.ndarray, adjacency: sp.csr_matrix, layers: int) -> np.ndarray:
    a_hat = symmetric_normalize(adjacency, dtype="float64")
    layer, pooled = e0, e0.copy()
    for _ in range(layers):
        layer = a_hat @ layer
        pooled = pooled + layer
    return pooled / (layers + 1)


def _bundle(e0: np.ndarray, adjacency: sp.csr_matrix, layers: int = 3) -> ServingBundle:
    return build_bundle(
        e0, adjacency, SPACE, layers, item_ids=np.arange(100, 100 + SPACE.n_item),
        alpha=ALPHA, lambda_decay=LAMBDA, d_day=DAY,
    )


# ── Do chinh xac ─────────────────────────────────────────────────────────


def test_fold_in_equals_full_propagation_for_a_train_visitor() -> None:
    """★ Visitor co trong graph, tau = T_train: fold-in == lan truyen toan do thi."""
    rng = np.random.default_rng(0)
    e0 = rng.normal(size=(SPACE.total, 8))
    adjacency, weights = _graph()
    z_full = _full_propagation(e0, adjacency, 3)
    bundle = _bundle(e0, adjacency)

    for visitor in HISTORY:
        z_fold = fold_in(bundle, weights[visitor], e0_user=e0[visitor],
                         weights_in_graph=weights[visitor])
        np.testing.assert_allclose(z_fold, z_full[visitor], rtol=1e-5, atol=1e-6)


def test_item_embeddings_in_the_bundle_equal_full_propagation() -> None:
    rng = np.random.default_rng(1)
    e0 = rng.normal(size=(SPACE.total, 8))
    adjacency, _ = _graph()
    z_full = _full_propagation(e0, adjacency, 3)
    np.testing.assert_allclose(
        _bundle(e0, adjacency).item_z, z_full[SPACE.item_slice()], rtol=1e-5, atol=1e-6
    )


def test_bundle_survives_a_save_load_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(2)
    adjacency, _ = _graph()
    bundle = _bundle(rng.normal(size=(SPACE.total, 4)), adjacency)
    path = tmp_path / "serving.npz"
    bundle.save(path)
    loaded = ServingBundle.load(path)
    for field in ("item_ids", "item_layers", "item_z", "item_degree", "alpha"):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(bundle, field))
    assert loaded.lambda_decay == LAMBDA and loaded.d_day == DAY


# ── Hanh vi theo thoi gian ───────────────────────────────────────────────


def test_events_after_tau_are_invisible() -> None:
    """Do thi tai tau khong duoc thay su kien xay ra sau tau."""
    events = pd.DataFrame({"item_idx": [0, 1], "behavior": ["view", "view"],
                           "timestamp": [T_TRAIN, T_TRAIN + DAY]})
    w = edge_weights_at(events, T_TRAIN, _skeleton())
    assert list(w.index) == [0]


def test_a_weight_decays_as_tau_moves_forward() -> None:
    """★ Cung mot su kien, tau cang xa thi trong so cang nho: 'dong' theo thoi gian."""
    events = pd.DataFrame({"item_idx": [0], "behavior": ["transaction"], "timestamp": [T_TRAIN]})
    now = edge_weights_at(events, T_TRAIN, _skeleton())[0]
    later = edge_weights_at(events, T_TRAIN + 30 * DAY, _skeleton())[0]
    assert now == 3.0
    np.testing.assert_allclose(later, 3.0 * np.exp(-LAMBDA * 30))


def test_a_new_click_changes_the_recommendations() -> None:
    """★ Hanh vi moi lam doi Top-K ngay, khong can train lai."""
    rng = np.random.default_rng(3)
    e0 = rng.normal(size=(SPACE.total, 8))
    adjacency, _ = _graph()
    bundle = _bundle(e0, adjacency)

    before = pd.DataFrame({"item_idx": [0], "behavior": ["view"], "timestamp": [T_TRAIN]})
    after = pd.concat([before, pd.DataFrame(
        {"item_idx": [3], "behavior": ["transaction"], "timestamp": [T_TRAIN]})])
    z_before = fold_in(bundle, edge_weights_at(before, T_TRAIN, bundle))
    z_after = fold_in(bundle, edge_weights_at(after, T_TRAIN, bundle))
    assert not np.allclose(z_before, z_after)

    top = recommend(bundle, after, T_TRAIN, k=3)
    assert not set(top["item_idx"]) & {0, 3}, "item da tuong tac phai bi loai"
    assert list(top["item_id"]) == [100 + i for i in top["item_idx"]]


# ── Xep hang ─────────────────────────────────────────────────────────────


def test_top_k_breaks_ties_by_ascending_item_index() -> None:
    """Quy tac D32: diem bang nhau thi item_idx nho hon dung truoc."""
    scores = np.array([0.5, 0.9, 0.5, 0.9, 0.1])
    assert list(top_k(scores, 3)) == [1, 3, 0]
    assert list(top_k(scores, 3, exclude=np.array([1]))) == [3, 0, 2]
