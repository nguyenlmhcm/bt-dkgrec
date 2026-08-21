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
    assert results["n_users"] == {"warm": 2, "cold": 1, "all": 3}


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
