"""Tests for the evaluator and the heuristic baselines (Buoc 5).

The evaluator owns the protocol, so these tests pin down the parts that would
otherwise differ silently between models: candidate scope, seen-filtering, the
conservative treatment of unrankable targets, and warm/cold separation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.mapping import IdMapping
from src.evaluation.evaluator import Evaluator, build_evaluation_set, build_seen_matrix
from src.guards.leakage import LeakageError
from src.models.base import ModelContext, NotFittedError, Recommender
from src.models.popularity import Popularity, RecentPopularity
from src.utils.config import load_config

DAY = 86_400_000
T_TRAIN = 100 * DAY


@pytest.fixture
def mapping() -> IdMapping:
    return IdMapping(
        visitor_ids=np.array([1, 2, 3], dtype="int32"),
        item_ids=np.array([10, 11, 12, 13], dtype="int32"),
    )


@pytest.fixture
def events() -> pd.DataFrame:
    """Train history plus a test period holding warm and cold targets."""
    rows = [
        # visitor, item, behavior, timestamp, split
        (1, 10, "view", 10 * DAY, "train"),
        (1, 11, "addtocart", 90 * DAY, "train"),
        (2, 11, "view", 20 * DAY, "train"),
        (2, 12, "transaction", 95 * DAY, "train"),
        (3, 13, "view", 30 * DAY, "train"),
        # test targets
        (1, 12, "addtocart", 110 * DAY, "test"),      # warm, rankable
        (2, 99, "transaction", 111 * DAY, "test"),    # warm, item not in train
        (4, 10, "addtocart", 112 * DAY, "test"),      # cold visitor
    ]
    frame = pd.DataFrame(rows, columns=["visitorid", "itemid", "behavior", "timestamp", "split"])
    frame["behavior"] = pd.Categorical(
        frame["behavior"], categories=["view", "addtocart", "transaction"]
    )
    frame["split"] = pd.Categorical(frame["split"], categories=["train", "valid", "test"])
    return frame


@pytest.fixture
def context(events: pd.DataFrame, mapping: IdMapping) -> ModelContext:
    cfg = load_config(model="popularity")
    train = events[events["split"] == "train"].copy()
    train["visitor_idx"] = mapping.visitor_index(train["visitorid"])
    train["item_idx"] = mapping.item_index(train["itemid"])
    return ModelContext(
        cfg=cfg, mapping=mapping, train_events=train,
        interaction_edges=pd.DataFrame(), t_train=T_TRAIN,
    )


def _indexed(events: pd.DataFrame, mapping: IdMapping) -> pd.DataFrame:
    out = events.copy()
    out["visitor_idx"] = mapping.visitor_index(out["visitorid"])
    out["item_idx"] = mapping.item_index(out["itemid"])
    return out


# ══ Evaluation set ═══════════════════════════════════════════════════════


def test_evaluation_set_separates_warm_from_cold(events, mapping) -> None:
    cfg = load_config(model="popularity")
    indexed = _indexed(events, mapping)
    seen = build_seen_matrix(indexed[indexed["split"] == "train"], mapping)
    eval_set = build_evaluation_set(indexed, mapping, "test", cfg, seen)

    assert eval_set.n_users == 3
    assert eval_set.is_warm.tolist() == [True, True, False]   # visitors 1, 2 warm; 4 cold


def test_target_outside_train_is_a_miss_but_stays_in_the_denominator(events, mapping) -> None:
    """Visitor 2's only target is item 99, which never appeared in train."""
    cfg = load_config(model="popularity")
    indexed = _indexed(events, mapping)
    seen = build_seen_matrix(indexed[indexed["split"] == "train"], mapping)
    eval_set = build_evaluation_set(indexed, mapping, "test", cfg, seen)

    position = int(np.flatnonzero(eval_set.visitor_ids == 2)[0])
    assert len(eval_set.relevant[position]) == 0     # nothing rankable
    assert eval_set.n_relevant_total[position] == 1  # ... but still counted
    assert eval_set.n_targets_unrankable["item_not_in_train"] == 1


def test_a_target_already_seen_in_train_is_filtered_but_still_counted(mapping) -> None:
    cfg = load_config(model="popularity")
    rows = [
        (1, 10, "view", 10 * DAY, "train"),
        (1, 10, "addtocart", 110 * DAY, "test"),   # re-engages an item seen in train
    ]
    frame = pd.DataFrame(rows, columns=["visitorid", "itemid", "behavior", "timestamp", "split"])
    frame["behavior"] = pd.Categorical(frame["behavior"], categories=["view", "addtocart", "transaction"])
    frame["split"] = pd.Categorical(frame["split"], categories=["train", "valid", "test"])
    indexed = _indexed(frame, mapping)
    seen = build_seen_matrix(indexed[indexed["split"] == "train"], mapping)

    eval_set = build_evaluation_set(indexed, mapping, "test", cfg, seen)
    assert len(eval_set.relevant[0]) == 0
    assert eval_set.n_relevant_total[0] == 1
    assert eval_set.n_targets_unrankable["removed_as_already_seen"] == 1


# ══ Ranking, candidate scope (rule 5) and seen-filtering ═════════════════


def test_ranking_never_recommends_an_item_the_user_already_saw(events, mapping, context) -> None:
    cfg = load_config(model="popularity")
    indexed = _indexed(events, mapping)
    seen = build_seen_matrix(indexed[indexed["split"] == "train"], mapping)
    model = Popularity()
    model.fit(context)

    eval_set = build_evaluation_set(indexed, mapping, "test", cfg, seen)
    rankings = Evaluator(cfg, mapping, seen).rank(model, eval_set)

    position = int(np.flatnonzero(eval_set.visitor_ids == 1)[0])
    recommended_ids = mapping.item_ids[rankings[position][rankings[position] >= 0]]
    assert 10 not in recommended_ids and 11 not in recommended_ids  # both seen in train


def test_rule5_guard_fires_when_a_model_ranks_outside_the_train_universe(events, mapping) -> None:
    """Leakage rule 5 is enforced on the produced ranking, not assumed."""
    cfg = load_config(model="popularity")
    indexed = _indexed(events, mapping)
    seen = build_seen_matrix(indexed[indexed["split"] == "train"], mapping)
    eval_set = build_evaluation_set(indexed, mapping, "test", cfg, seen)

    evaluator = Evaluator(cfg, mapping, seen)
    # A mapping that claims fewer train items than the ranking will reference.
    evaluator.mapping = IdMapping(
        visitor_ids=mapping.visitor_ids, item_ids=np.array([10, 11], dtype="int32")
    )

    class WideModel(Recommender):
        name, supports_cold_start = "wide", True

        def fit(self, context: ModelContext) -> None: ...

        def score(self, visitor_indices: np.ndarray) -> np.ndarray:
            return np.tile(np.array([0.1, 0.2, 0.3, 0.4]), (len(visitor_indices), 1))

    with pytest.raises((LeakageError, IndexError)):
        evaluator.rank(WideModel(), eval_set)


# ══ Segments ═════════════════════════════════════════════════════════════


def test_personalised_model_reports_null_for_cold_and_all(events, mapping, context) -> None:
    """CLAUDE.md: never mix cold users into a personalised model's metric."""
    cfg = load_config(model="popularity")
    indexed = _indexed(events, mapping)
    seen = build_seen_matrix(indexed[indexed["split"] == "train"], mapping)
    eval_set = build_evaluation_set(indexed, mapping, "test", cfg, seen)

    class PersonalisedModel(Popularity):
        name, supports_cold_start = "personalised", False

    model = PersonalisedModel()
    model.fit(context)
    results, _ = Evaluator(cfg, mapping, seen).evaluate(model, eval_set)

    assert results["warm"] is not None
    assert results["cold"] is None      # not zero -- unmeasured, not measured as 0
    assert results["all"] is None
    assert results["n_users"]["warm"] == 2
    assert results["n_users"]["cold"] == 1
    assert results["n_users"]["all"] == 3


# ══ Phan tang theo bac (Buoc 7 bis) ══════════════════════════════════════
#
# W(u,i) is aggregated per (visitor, item) edge and the LightGCN normalisation
# divides a visitor's row by that visitor's own weight total. For a visitor with
# a single edge the weight is divided by itself and cancels exactly, so the
# dynamic and the static knowledge graph are provably identical for them.
# Reporting the bands separately is what lets the thesis measure the mechanism
# where it is free to act rather than diluting it across a population where it
# mathematically cannot.


def _degree_fixture() -> tuple[IdMapping, pd.DataFrame]:
    """Warm visitors of degree 1, 2 and 3, plus a cold one."""
    mapping = IdMapping(
        visitor_ids=np.array([1, 2, 3], dtype="int32"),
        item_ids=np.array([10, 11, 12, 13], dtype="int32"),
    )
    rows = [
        (1, 10, "view", 10 * DAY, "train"),                    # visitor 1: 1 canh
        (2, 10, "view", 10 * DAY, "train"),                    # visitor 2: 2 canh
        (2, 11, "view", 11 * DAY, "train"),
        (3, 10, "view", 10 * DAY, "train"),                    # visitor 3: 3 canh
        (3, 11, "view", 11 * DAY, "train"),
        (3, 12, "view", 12 * DAY, "train"),
        # Repeat events on an existing edge must NOT raise the degree.
        (1, 10, "addtocart", 20 * DAY, "train"),
        (1, 13, "addtocart", 110 * DAY, "test"),
        (2, 13, "addtocart", 110 * DAY, "test"),
        (3, 13, "addtocart", 110 * DAY, "test"),
        (9, 13, "addtocart", 110 * DAY, "test"),               # cold
    ]
    frame = pd.DataFrame(rows, columns=["visitorid", "itemid", "behavior", "timestamp", "split"])
    frame["behavior"] = pd.Categorical(
        frame["behavior"], categories=["view", "addtocart", "transaction"]
    )
    frame["split"] = pd.Categorical(frame["split"], categories=["train", "valid", "test"])
    return mapping, frame


def _degree_eval_set():
    mapping, frame = _degree_fixture()
    cfg = load_config(model="popularity")
    indexed = _indexed(frame, mapping)
    seen = build_seen_matrix(indexed[indexed["split"] == "train"], mapping)
    return cfg, mapping, seen, build_evaluation_set(indexed, mapping, "test", cfg, seen)


def test_train_degree_counts_edges_not_events() -> None:
    """Visitor 1 has two train events on one item -- that is ONE edge."""
    _, _, _, eval_set = _degree_eval_set()

    by_visitor = dict(zip(eval_set.visitor_ids.tolist(), eval_set.train_degree.tolist()))
    assert by_visitor == {1: 1, 2: 2, 3: 3, 9: 0}


def test_a_cold_visitor_has_degree_zero_and_joins_no_band() -> None:
    _, _, _, eval_set = _degree_eval_set()

    position = int(np.flatnonzero(eval_set.visitor_ids == 9)[0])
    assert eval_set.train_degree[position] == 0
    assert not eval_set.is_warm[position]


def test_degree_bands_partition_the_warm_segment_exactly() -> None:
    """No warm user may be counted twice, and none may be lost."""
    cfg, mapping, seen, eval_set = _degree_eval_set()
    model = Popularity()
    model.fit(ModelContext(
        cfg=cfg, mapping=mapping,
        train_events=_indexed(_degree_fixture()[1], mapping).query("split == 'train'"),
        interaction_edges=pd.DataFrame(), t_train=T_TRAIN,
    ))
    results, _ = Evaluator(cfg, mapping, seen).evaluate(model, eval_set)

    counts = results["n_users"]
    bands = counts["warm_deg1"] + counts["warm_deg2"] + counts["warm_deg3plus"]
    assert bands == counts["warm"] == 3
    assert (counts["warm_deg1"], counts["warm_deg2"], counts["warm_deg3plus"]) == (1, 1, 1)


def test_degree_bands_are_measured_for_a_personalised_model() -> None:
    """They are warm subsets, so the cold-start rule must not null them out."""
    cfg, mapping, seen, eval_set = _degree_eval_set()

    class PersonalisedModel(Popularity):
        name, supports_cold_start = "personalised", False

    model = PersonalisedModel()
    model.fit(ModelContext(
        cfg=cfg, mapping=mapping,
        train_events=_indexed(_degree_fixture()[1], mapping).query("split == 'train'"),
        interaction_edges=pd.DataFrame(), t_train=T_TRAIN,
    ))
    results, _ = Evaluator(cfg, mapping, seen).evaluate(model, eval_set)

    for band in ("warm_deg1", "warm_deg2", "warm_deg3plus"):
        assert results[band] is not None, f"{band} bi bao cao la None"
    assert results["cold"] is None       # the real cold rule still holds


def test_an_empty_band_is_null_rather_than_zero(events, mapping, context) -> None:
    """Same honesty rule as cold: unmeasured is never reported as 0."""
    cfg = load_config(model="popularity")
    indexed = _indexed(events, mapping)
    seen = build_seen_matrix(indexed[indexed["split"] == "train"], mapping)
    eval_set = build_evaluation_set(indexed, mapping, "test", cfg, seen)

    model = Popularity()
    model.fit(context)
    results, _ = Evaluator(cfg, mapping, seen).evaluate(model, eval_set)

    assert results["n_users"]["warm_deg1"] == 0
    assert results["warm_deg1"] is None


def test_popularity_can_serve_cold_users(events, mapping, context) -> None:
    cfg = load_config(model="popularity")
    indexed = _indexed(events, mapping)
    seen = build_seen_matrix(indexed[indexed["split"] == "train"], mapping)
    eval_set = build_evaluation_set(indexed, mapping, "test", cfg, seen)

    model = Popularity()
    model.fit(context)
    results, _ = Evaluator(cfg, mapping, seen).evaluate(model, eval_set)

    assert results["cold"] is not None
    assert results["all"] is not None


# ══ Baselines ════════════════════════════════════════════════════════════


def test_popularity_ranks_by_target_behavior_count(context) -> None:
    model = Popularity()
    model.fit(context)
    scores = model.score(np.array([0]))[0]
    # train targets: item 11 addtocart (idx 1), item 12 transaction (idx 2)
    assert scores[1] == 1.0 and scores[2] == 1.0
    assert scores[0] == 0.0 and scores[3] == 0.0   # views only


def test_popularity_is_identical_for_every_visitor(context) -> None:
    model = Popularity()
    model.fit(context)
    scores = model.score(np.array([0, 1, 2]))
    assert np.array_equal(scores[0], scores[1]) and np.array_equal(scores[1], scores[2])


def test_popularity_refuses_to_score_before_fit() -> None:
    with pytest.raises(NotFittedError):
        Popularity().score(np.array([0]))


def test_recent_popularity_only_counts_the_recent_window(events, mapping) -> None:
    cfg = load_config(model="recent_popularity", overrides={"model": {"recent_window_days": 20}})
    train = _indexed(events, mapping)
    train = train[train["split"] == "train"]
    context = ModelContext(
        cfg=cfg, mapping=mapping, train_events=train,
        interaction_edges=pd.DataFrame(), t_train=T_TRAIN,
    )
    model = RecentPopularity()
    model.fit(context)
    scores = model.score(np.array([0]))[0]

    # Window is (80*DAY, 100*DAY]: item 11 at 90*DAY and item 12 at 95*DAY.
    assert scores[1] == 1.0 and scores[2] == 1.0
    assert scores.sum() == 2.0


def test_recent_popularity_inherits_popularity() -> None:
    assert issubclass(RecentPopularity, Popularity)
    overridden = {
        name for name, value in vars(RecentPopularity).items()
        if callable(value) and not name.startswith("__")
    }
    assert overridden == {"_select_events", "describe"}


def test_baselines_are_deterministic_so_seed_std_is_zero(context) -> None:
    """Footnote requirement D8: std = 0 is a property, not a bug."""
    first, second = Popularity(), Popularity()
    first.fit(context)
    second.fit(context)
    assert np.array_equal(first.score(np.array([0])), second.score(np.array([0])))


# ══ Tie-breaking ═════════════════════════════════════════════════════════
#
# Popularity scores are interaction counts, so ties are the rule rather than
# the exception: on cohort Original, 205.106 items share 73 distinct scores.
# Without a fixed rule the winner at the Top-K boundary is whatever the
# compiled kernel happens to pick, and the same run measures differently on
# two machines.


def _evaluator(mapping: IdMapping, n_items: int, k_values=(2, 3)) -> Evaluator:
    cfg = load_config(
        model="popularity",
        overrides={"evaluation": {"k_values": list(k_values), "primary_k": max(k_values)}},
    )
    seen = build_seen_matrix(
        pd.DataFrame({"visitor_idx": [], "item_idx": []}, dtype="int64"), mapping
    )
    return Evaluator(cfg, mapping, seen)


def _mapping_of(n_items: int) -> IdMapping:
    return IdMapping(
        visitor_ids=np.array([1], dtype="int32"),
        item_ids=np.arange(10, 10 + n_items, dtype="int32"),
    )


def test_ties_are_broken_by_ascending_item_index() -> None:
    """★ Equal scores: the smaller index wins, on every machine."""
    mapping = _mapping_of(6)
    evaluator = _evaluator(mapping, 6)
    #                 idx: 0    1    2    3    4    5
    scores = np.array([[1.0, 9.0, 5.0, 9.0, 5.0, 5.0]], dtype="float32")

    ranked = evaluator._top_k_indices(scores)

    # 9 at indices 1 and 3 -> 1 first. 5 at 2, 4, 5 -> 2 first.
    assert ranked[0].tolist() == [1, 3, 2]


def test_the_contested_boundary_admits_the_smaller_index() -> None:
    """The real shape of the bug: two items tied exactly at rank K."""
    mapping = _mapping_of(5)
    evaluator = _evaluator(mapping, 5, k_values=(2,))
    #                 idx: 0    1    2    3    4
    scores = np.array([[3.0, 7.0, 9.0, 7.0, 1.0]], dtype="float32")

    ranked = evaluator._top_k_indices(scores)

    # Rank 1 is index 2 (score 9). Rank 2 is contested between 1 and 3, both
    # scoring 7 -- exactly the situation that made Colab and the VPS disagree.
    assert ranked[0].tolist() == [2, 1]


def test_ranking_matches_a_brute_force_reference_on_heavily_tied_scores() -> None:
    """Property check against an independent, obviously-correct implementation."""
    rng = np.random.default_rng(2020)
    n_rows, n_items, k = 40, 200, 20
    mapping = _mapping_of(n_items)
    evaluator = _evaluator(mapping, n_items, k_values=(k,))
    # Integers in a narrow range: about ten items share every score.
    scores = rng.integers(0, 20, size=(n_rows, n_items)).astype("float32")

    ranked = evaluator._top_k_indices(scores)

    for row in range(n_rows):
        # lexsort: last key is primary. Sort by score descending, index ascending.
        reference = np.lexsort((np.arange(n_items), -scores[row]))[:k]
        assert ranked[row].tolist() == reference.tolist()


def test_filtered_items_sort_last_rather_than_first() -> None:
    """Seen-filtering writes -inf; negating for the sort must not float it up."""
    mapping = _mapping_of(4)
    evaluator = _evaluator(mapping, 4, k_values=(3,))
    scores = np.array([[-np.inf, 2.0, -np.inf, 8.0]], dtype="float32")

    ranked = evaluator._top_k_indices(scores)

    assert ranked[0].tolist()[:2] == [3, 1]
    assert np.isneginf(scores[0, ranked[0, 2]])


def test_ranking_is_unchanged_when_the_same_row_is_scored_in_a_different_batch() -> None:
    """Row independence: batching is infrastructure and must not move a result."""
    mapping = _mapping_of(50)
    evaluator = _evaluator(mapping, 50, k_values=(10,))
    rng = np.random.default_rng(7)
    scores = rng.integers(0, 5, size=(8, 50)).astype("float32")

    together = evaluator._top_k_indices(scores)
    alone = np.vstack([evaluator._top_k_indices(scores[row : row + 1]) for row in range(8)])

    assert np.array_equal(together, alone)


def test_ranking_handles_a_universe_smaller_than_k() -> None:
    mapping = _mapping_of(3)
    evaluator = _evaluator(mapping, 3, k_values=(10,))
    scores = np.array([[4.0, 4.0, 9.0]], dtype="float32")

    ranked = evaluator._top_k_indices(scores)

    assert ranked[0].tolist() == [2, 0, 1]


def test_every_item_tied_at_the_same_score_still_yields_k_items() -> None:
    """The degenerate case: 90% of Original's items are tied at zero."""
    mapping = _mapping_of(100)
    evaluator = _evaluator(mapping, 100, k_values=(20,))
    scores = np.zeros((3, 100), dtype="float32")

    ranked = evaluator._top_k_indices(scores)

    assert ranked.shape == (3, 20)
    for row in range(3):
        assert ranked[row].tolist() == list(range(20))
