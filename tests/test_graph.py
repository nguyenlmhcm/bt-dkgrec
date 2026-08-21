"""Tests for the graph layer (Buoc 4).

The load-bearing test of the whole thesis is
:func:`test_ablation_pair_differs_only_in_edge_weights`: ``bt_dkgrec`` and
``static_kg_gcn`` must produce graphs with an identical sparsity pattern and
different values. If that ever stops holding, the ablation stops being a
controlled comparison and the central claim loses its evidence.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.graph.builder import build_graph
from src.graph.normalize import assert_symmetric, symmetric_normalize, weighted_degree
from src.graph.schema import NodeSpace, RelationType
from src.graph.weighting import (
    BehaviorTimeWeighting,
    UniformWeighting,
    aggregate_interaction_edges,
    event_age_days,
    weighting_for_model,
)
from src.utils.config import load_config

DAY = 86_400_000
T_TRAIN = 100 * DAY
BEHAVIORS = ("view", "addtocart", "transaction")
ALPHA = np.array([1.0, 2.0, 3.0])
LAMBDA = 0.01


@pytest.fixture
def bt_weighting() -> BehaviorTimeWeighting:
    return BehaviorTimeWeighting(alpha=ALPHA, lambda_decay=LAMBDA, behaviors=BEHAVIORS)


# ══ Formula (3.16) — event age ═══════════════════════════════════════════


def test_event_age_is_measured_in_days() -> None:
    ages = event_age_days(np.array([100 * DAY, 90 * DAY, 50 * DAY]), T_TRAIN, DAY)
    assert ages.tolist() == [0.0, 10.0, 50.0]


def test_event_age_clips_at_zero() -> None:
    """An event on or past the boundary decays by nothing, never gains weight."""
    ages = event_age_days(np.array([T_TRAIN + 5 * DAY]), T_TRAIN, DAY)
    assert ages.tolist() == [0.0]


# ══ Formula (3.17) — event weight ════════════════════════════════════════


def test_behavior_time_weight_matches_the_formula(bt_weighting: BehaviorTimeWeighting) -> None:
    codes = np.array([0, 1, 2])                 # view, addtocart, transaction
    delta = np.array([0.0, 10.0, 50.0])
    weights = bt_weighting.edge_weight(codes, delta)

    assert weights[0] == pytest.approx(1.0 * math.exp(-0.01 * 0.0))
    assert weights[1] == pytest.approx(2.0 * math.exp(-0.01 * 10.0))
    assert weights[2] == pytest.approx(3.0 * math.exp(-0.01 * 50.0))


def test_stronger_behavior_outweighs_weaker_at_equal_age(bt_weighting) -> None:
    delta = np.array([7.0, 7.0, 7.0])
    weights = bt_weighting.edge_weight(np.array([0, 1, 2]), delta)
    assert weights[0] < weights[1] < weights[2]


def test_older_event_outweighed_by_newer_of_same_behavior(bt_weighting) -> None:
    weights = bt_weighting.edge_weight(np.array([1, 1]), np.array([1.0, 60.0]))
    assert weights[0] > weights[1] > 0


def test_weights_are_always_strictly_positive(bt_weighting) -> None:
    weights = bt_weighting.edge_weight(np.array([0, 1, 2] * 10), np.linspace(0, 500, 30))
    assert (weights > 0).all()


def test_invalid_weighting_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        BehaviorTimeWeighting(np.array([1.0, 0.0, 3.0]), LAMBDA, BEHAVIORS)
    with pytest.raises(ValueError, match="lambda"):
        BehaviorTimeWeighting(ALPHA, -0.5, BEHAVIORS)


# ══ Ablation: override sach ══════════════════════════════════════════════


def test_uniform_weighting_returns_one_regardless_of_behavior_or_age() -> None:
    uniform = UniformWeighting(alpha=ALPHA, lambda_decay=LAMBDA, behaviors=BEHAVIORS)
    weights = uniform.edge_weight(np.array([0, 1, 2]), np.array([0.0, 10.0, 500.0]))
    assert weights.tolist() == [1.0, 1.0, 1.0]


def test_uniform_weighting_overrides_exactly_one_computational_method() -> None:
    """CLAUDE.md: the ablation may differ in edge_weight() and nothing else."""
    assert issubclass(UniformWeighting, BehaviorTimeWeighting)
    overridden = {
        name for name, value in vars(UniformWeighting).items()
        if callable(value) and not name.startswith("__")
    }
    assert overridden == {"edge_weight", "describe"}, overridden
    # describe() is metadata for graph_stats.json, not part of the computation.
    assert UniformWeighting.__init__ is BehaviorTimeWeighting.__init__


def test_weighting_registry_maps_each_model_to_its_strategy() -> None:
    assert type(weighting_for_model(load_config(model="bt_dkgrec"))) is BehaviorTimeWeighting
    assert type(weighting_for_model(load_config(model="static_kg_gcn"))) is UniformWeighting
    assert type(weighting_for_model(load_config(model="lightgcn"))) is UniformWeighting


def test_heuristic_models_have_no_graph() -> None:
    with pytest.raises(ValueError, match="khong dung do thi"):
        weighting_for_model(load_config(model="popularity"))


def test_alpha_and_lambda_come_from_config_not_from_code() -> None:
    strategy = weighting_for_model(load_config(model="bt_dkgrec"))
    described = strategy.describe()
    assert described["alpha"] == {"view": 1.0, "addtocart": 2.0, "transaction": 3.0}
    assert described["lambda_decay"] == 0.01


# ══ Formula (3.18) — aggregated edge weight ══════════════════════════════


def test_aggregated_edge_weight_sums_every_event(bt_weighting) -> None:
    events = pd.DataFrame(
        {
            "visitor_idx": [0, 0, 0, 1],
            "item_idx": [0, 0, 1, 2],
            "behavior": ["view", "addtocart", "view", "transaction"],
            "timestamp": [100 * DAY, 90 * DAY, 50 * DAY, 100 * DAY],
        }
    )
    edges = aggregate_interaction_edges(
        events, bt_weighting, T_TRAIN, DAY, BEHAVIORS, ("addtocart", "transaction")
    )
    pair = edges[(edges["visitor_idx"] == 0) & (edges["item_idx"] == 0)].iloc[0]

    expected = 1.0 * math.exp(0.0) + 2.0 * math.exp(-0.01 * 10.0)
    assert pair["weight"] == pytest.approx(expected)
    assert (pair["n_view"], pair["n_cart"], pair["n_txn"]) == (1, 1, 0)
    assert pair["last_ts"] == 100 * DAY
    assert len(edges) == 3


def test_aggregation_rejects_an_unknown_behavior(bt_weighting) -> None:
    events = pd.DataFrame(
        {
            "visitor_idx": [0], "item_idx": [0],
            "behavior": ["wishlist"], "timestamp": [100 * DAY],
        }
    )
    with pytest.raises(ValueError, match="behavior khong nam trong cau hinh"):
        aggregate_interaction_edges(
            events, bt_weighting, T_TRAIN, DAY, BEHAVIORS, ("addtocart",)
        )


def test_uniform_aggregation_equals_the_event_count() -> None:
    """With w == 1 the aggregated weight is simply how many events occurred."""
    uniform = UniformWeighting(alpha=ALPHA, lambda_decay=LAMBDA, behaviors=BEHAVIORS)
    events = pd.DataFrame(
        {
            "visitor_idx": [0, 0, 0],
            "item_idx": [0, 0, 0],
            "behavior": ["view", "view", "transaction"],
            "timestamp": [10 * DAY, 20 * DAY, 30 * DAY],
        }
    )
    edges = aggregate_interaction_edges(
        events, uniform, T_TRAIN, DAY, BEHAVIORS, ("transaction",)
    )
    assert edges.iloc[0]["weight"] == pytest.approx(3.0)


# ══ Normalisation (3.24)-(3.26) ══════════════════════════════════════════


def test_symmetric_normalisation_matches_hand_computation() -> None:
    adjacency = sp.csr_matrix(np.array([[0.0, 4.0], [4.0, 0.0]]))
    normalized = symmetric_normalize(adjacency).toarray()
    # d = 4 for both -> 4 / (sqrt(4) * sqrt(4)) = 1
    assert normalized == pytest.approx(np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_normalisation_damps_a_high_degree_node() -> None:
    """A hub's edges are scaled down relative to a low-degree node's edge."""
    adjacency = sp.csr_matrix(
        np.array([[0.0, 1.0, 1.0, 1.0], [1.0, 0, 0, 0], [1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    )
    normalized = symmetric_normalize(adjacency).toarray()
    assert normalized[0, 1] == pytest.approx(1 / math.sqrt(3))
    assert normalized[0, 1] < 1.0


def test_normalisation_survives_an_isolated_node() -> None:
    adjacency = sp.csr_matrix(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    normalized = symmetric_normalize(adjacency)
    assert np.isfinite(normalized.toarray()).all()
    assert weighted_degree(adjacency)[2] == 0.0


def test_normalisation_rejects_a_negative_weight() -> None:
    with pytest.raises(ValueError, match="trong so am"):
        symmetric_normalize(sp.csr_matrix(np.array([[0.0, -1.0], [-1.0, 0.0]])))


def test_normalisation_rejects_a_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="phai vuong"):
        symmetric_normalize(sp.csr_matrix(np.zeros((2, 3))))


def test_assert_symmetric_catches_a_one_way_edge() -> None:
    asymmetric = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="khong doi xung"):
        assert_symmetric(asymmetric)


# ══ Node space ═══════════════════════════════════════════════════════════


def test_node_space_blocks_are_contiguous_and_ordered() -> None:
    space = NodeSpace(n_visitor=10, n_item=5, n_category=3, n_property_value=2)
    assert (space.visitor_offset, space.item_offset) == (0, 10)
    assert (space.category_offset, space.property_value_offset) == (15, 18)
    assert space.total == 20
    assert space.item_slice() == slice(10, 15)
    assert space.item_ids(np.array([0, 4])).tolist() == [10, 14]


def test_node_space_without_side_info_has_two_blocks() -> None:
    space = NodeSpace(n_visitor=10, n_item=5)
    assert space.total == 15
    assert space.category_offset == space.property_value_offset == 15


# ══ Integration: dung graph tu artefact tong hop ═════════════════════════


@pytest.fixture
def interim_dir(tmp_path):
    """A tiny but complete `data/interim/<cohort>/` directory."""
    events = pd.DataFrame(
        {
            "timestamp": np.array(
                [100 * DAY, 90 * DAY, 50 * DAY, 100 * DAY, 110 * DAY], dtype="int64"
            ),
            "visitorid": np.array([1, 1, 1, 2, 1], dtype="int32"),
            "itemid": np.array([10, 10, 11, 12, 13], dtype="int32"),
            "behavior": pd.Categorical(
                ["view", "addtocart", "view", "transaction", "view"], categories=list(BEHAVIORS)
            ),
            "split": pd.Categorical(
                ["train"] * 4 + ["test"], categories=["train", "valid", "test"]
            ),
        }
    )
    events.to_parquet(tmp_path / "events.parquet", index=False)
    pd.DataFrame({"visitor_id": [1, 2], "idx": [0, 1]}).to_parquet(
        tmp_path / "visitors.parquet", index=False
    )
    pd.DataFrame({"item_id": [10, 11, 12], "idx": [0, 1, 2]}).to_parquet(
        tmp_path / "items.parquet", index=False
    )
    pd.DataFrame(
        {
            "item_idx": np.array([0, 1, 2], dtype="int32"),
            "category_id": np.array([100, 100, 100], dtype="int32"),
            "category_idx": np.array([0, 0, 0], dtype="int32"),
            "valid_from": np.array([1, 2, 3], dtype="int64"),
        }
    ).to_parquet(tmp_path / "side_item_category.parquet", index=False)
    pd.DataFrame(
        {
            "item_idx": np.array([0, 1, 2], dtype="int32"),
            "pv_idx": np.array([0, 0, 1], dtype="int32"),
            "valid_from": np.array([1, 2, 3], dtype="int64"),
        }
    ).to_parquet(tmp_path / "side_item_property.parquet", index=False)
    pd.DataFrame(
        {"pv_idx": [0, 1], "prop_key": ["a", "b"], "prop_value": ["x", "y"],
         "freq": [2, 1], "pv_id": ["a::x", "b::y"]}
    ).to_parquet(tmp_path / "side_property_values.parquet", index=False)
    pd.DataFrame(
        {"category_id": [100], "idx": [0], "depth": [0], "is_root": [True]}
    ).to_parquet(tmp_path / "side_categories.parquet", index=False)
    pd.DataFrame(
        {"category_idx": np.array([], dtype="int32"), "parent_idx": np.array([], dtype="int32")}
    ).to_parquet(tmp_path / "side_category_parent.parquet", index=False)
    (tmp_path / "split.json").write_text(
        json.dumps({"t_min": 0, "t_max": 120 * DAY, "t_train": T_TRAIN,
                    "t_valid_end": 105 * DAY, "boundary_inclusive": True}),
        encoding="utf-8",
    )
    return tmp_path


def test_build_graph_uses_train_events_only(interim_dir) -> None:
    cfg = load_config(model="bt_dkgrec")
    graph, edges = build_graph(cfg, interim_dir, weighting_for_model(cfg))
    # The test-split event on item 13 must not become an edge.
    assert graph.edge_counts[RelationType.INTERACTED_WITH.value] == 3
    assert edges["last_ts"].max() <= T_TRAIN


def test_build_graph_produces_a_symmetric_graph_without_isolated_nodes(interim_dir) -> None:
    cfg = load_config(model="bt_dkgrec")
    graph, _ = build_graph(cfg, interim_dir, weighting_for_model(cfg))
    assert_symmetric(graph.adjacency)
    assert graph.stats["isolated_nodes"] == 0
    assert graph.adjacency.nnz == 2 * graph.n_edges  # no duplicate edges collapsed


def test_lightgcn_graph_drops_the_side_information_blocks(interim_dir) -> None:
    cfg = load_config(model="lightgcn")
    graph, _ = build_graph(cfg, interim_dir, weighting_for_model(cfg))
    assert graph.node_space.n_category == 0
    assert graph.node_space.n_property_value == 0
    assert set(graph.edge_counts) == {RelationType.INTERACTED_WITH.value}


def test_ablation_pair_differs_only_in_edge_weights(interim_dir) -> None:
    """★ The central claim of the thesis, as an executable assertion.

    bt_dkgrec and static_kg_gcn must share node space, edge set and sparsity
    pattern exactly, and differ only in the numeric weights.
    """
    dynamic_cfg = load_config(model="bt_dkgrec")
    static_cfg = load_config(model="static_kg_gcn")
    dynamic, _ = build_graph(dynamic_cfg, interim_dir, weighting_for_model(dynamic_cfg))
    static, _ = build_graph(static_cfg, interim_dir, weighting_for_model(static_cfg))

    assert dynamic.node_space == static.node_space
    assert dynamic.edge_counts == static.edge_counts

    a, b = dynamic.adjacency.tocsr(), static.adjacency.tocsr()
    assert np.array_equal(a.indptr, b.indptr)      # identical sparsity pattern
    assert np.array_equal(a.indices, b.indices)
    assert not np.allclose(a.data, b.data)         # ... but different weights

    assert dynamic.stats["weighting"] == "behavior_time"
    assert static.stats["weighting"] == "uniform"
