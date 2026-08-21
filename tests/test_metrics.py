"""Tests for the ranking metrics (Buoc 5).

Metrics are defined once and every model reports through them, so a defect here
would silently corrupt every number in the thesis. Each metric is checked
against a hand-computed value rather than against another implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_metrics,
    coverage_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
)


# ══ Recall ═══════════════════════════════════════════════════════════════


def test_recall_counts_hits_over_all_targets() -> None:
    hits = np.array([True, False, False, True, False])
    assert recall_at_k(hits, n_relevant=4, k=5) == pytest.approx(0.5)


def test_recall_respects_the_cut_off() -> None:
    hits = np.array([False, False, True])
    assert recall_at_k(hits, n_relevant=1, k=2) == 0.0
    assert recall_at_k(hits, n_relevant=1, k=3) == 1.0


def test_recall_denominator_includes_unrankable_targets() -> None:
    """Conservative policy: a target outside I_train stays in the denominator."""
    hits = np.array([True, False])
    # One target retrieved, but the user actually had four targets, two of
    # which never appeared in train and therefore could not be ranked.
    assert recall_at_k(hits, n_relevant=4, k=2) == pytest.approx(0.25)


def test_recall_of_a_user_with_no_targets_is_zero() -> None:
    assert recall_at_k(np.array([False]), n_relevant=0, k=1) == 0.0


# ══ HitRate ══════════════════════════════════════════════════════════════


def test_hit_rate_is_binary() -> None:
    assert hit_rate_at_k(np.array([False, True, False]), k=3) == 1.0
    assert hit_rate_at_k(np.array([False, False, True]), k=2) == 0.0


# ══ NDCG ═════════════════════════════════════════════════════════════════


def test_ndcg_of_a_perfect_ranking_is_one() -> None:
    hits = np.array([True, True, False, False])
    assert ndcg_at_k(hits, n_relevant=2, k=4) == pytest.approx(1.0)


def test_ndcg_matches_hand_computation() -> None:
    hits = np.array([False, True, False])          # single hit at rank 2
    dcg = 1 / math.log2(3)                          # 1/log2(rank+2), rank index 1
    idcg = 1 / math.log2(2) + 1 / math.log2(3)      # two ideal targets
    assert ndcg_at_k(hits, n_relevant=2, k=3) == pytest.approx(dcg / idcg)


def test_ndcg_rewards_a_higher_rank() -> None:
    early = ndcg_at_k(np.array([True, False, False]), n_relevant=1, k=3)
    late = ndcg_at_k(np.array([False, False, True]), n_relevant=1, k=3)
    assert early > late > 0


def test_ndcg_is_zero_when_every_target_is_unrankable() -> None:
    assert ndcg_at_k(np.array([False, False]), n_relevant=3, k=2) == 0.0


def test_ndcg_ideal_list_is_capped_at_k() -> None:
    """A user with 100 targets cannot be penalised for a list only 10 long."""
    hits = np.array([True] * 10)
    assert ndcg_at_k(hits, n_relevant=100, k=10) == pytest.approx(1.0)


# ══ Coverage ═════════════════════════════════════════════════════════════


def test_coverage_counts_distinct_items_over_the_train_catalogue() -> None:
    top_k = [np.array([1, 2, 3]), np.array([2, 3, 4])]
    assert coverage_at_k(top_k, n_train_items=100, k=3) == pytest.approx(4 / 100)


def test_coverage_respects_the_cut_off() -> None:
    top_k = [np.array([1, 2, 3, 4])]
    assert coverage_at_k(top_k, n_train_items=10, k=2) == pytest.approx(2 / 10)


def test_coverage_of_an_identical_list_for_everyone_is_tiny() -> None:
    """A non-personalised model recommends the same items to all users."""
    top_k = [np.array([1, 2])] * 500
    assert coverage_at_k(top_k, n_train_items=200_000, k=2) == pytest.approx(2 / 200_000)


# ══ Aggregation ══════════════════════════════════════════════════════════


def test_compute_metrics_averages_over_users() -> None:
    results = compute_metrics(
        top_k_items=[np.array([10, 11]), np.array([12, 13])],
        relevant_sets=[np.array([10]), np.array([99])],   # user 1 hits, user 2 misses
        n_relevant_total=[1, 1],
        k_values=[2],
        n_train_items=100,
    )
    assert results["recall@2"] == pytest.approx(0.5)
    assert results["hit_rate@2"] == pytest.approx(0.5)
    assert results["coverage@2"] == pytest.approx(4 / 100)


def test_compute_metrics_reports_every_requested_cut_off() -> None:
    results = compute_metrics(
        top_k_items=[np.arange(20)],
        relevant_sets=[np.array([15])],
        n_relevant_total=[1],
        k_values=[10, 20],
        n_train_items=50,
    )
    assert results["recall@10"] == 0.0      # item 15 sits beyond rank 10
    assert results["recall@20"] == 1.0
    assert set(results) == {
        f"{m}@{k}" for k in (10, 20) for m in ("recall", "ndcg", "hit_rate", "coverage")
    }


def test_compute_metrics_on_an_empty_user_set_returns_zeros() -> None:
    results = compute_metrics([], [], [], k_values=[10], n_train_items=100)
    assert results == {"recall@10": 0.0, "ndcg@10": 0.0, "hit_rate@10": 0.0, "coverage@10": 0.0}


def test_compute_metrics_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="cung do dai"):
        compute_metrics([np.array([1])], [], [1], k_values=[1], n_train_items=10)
