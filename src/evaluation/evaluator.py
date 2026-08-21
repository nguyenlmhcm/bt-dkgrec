"""Offline Top-K evaluation, shared by every model.

The evaluator owns the whole protocol -- candidate scope, seen-filtering, top-K
selection, segmentation and metrics -- so that switching model changes nothing
else. Models only supply scores (:mod:`src.models.base`).

Protocol, as specified in CLAUDE.md:

* Candidate set is ``I_train``. Leakage rule 5 is asserted **here**, on the
  produced ranking, because this is the first step where a candidate artefact
  actually exists (docs/DECISIONS.md muc D14).
* Items the visitor already interacted with in train are removed from the
  recommendation list.
* A target outside ``I_train`` -- or removed by seen-filtering -- cannot be
  ranked. It counts as a miss and stays in the denominator.
* Warm and cold visitors are reported separately and never mixed into one
  metric for a personalised model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data.mapping import IdMapping
from src.evaluation.metrics import compute_metrics
from src.guards.leakage import assert_candidate_scope
from src.models.base import Recommender
from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)

SEGMENTS = ("warm", "cold", "all")


@dataclass(frozen=True)
class EvaluationSet:
    """Ground truth for one evaluation split.

    Attributes:
        split: ``"valid"`` or ``"test"``.
        visitor_ids: Raw visitor ids, one per evaluated user.
        visitor_indices: Matrix index per user, ``-1`` when cold.
        relevant: Rankable target item indices per user.
        n_relevant_total: Full target count per user, including targets that
            cannot be ranked. Denominator of Recall and NDCG.
        is_warm: Whether each user exists in the train mapping.
        n_targets_unrankable: How many targets were unrankable overall, split by
            reason -- for the honesty note in the report.
    """

    split: str
    visitor_ids: np.ndarray
    visitor_indices: np.ndarray
    relevant: list[np.ndarray]
    n_relevant_total: list[int]
    is_warm: np.ndarray
    n_targets_unrankable: dict[str, int]

    @property
    def n_users(self) -> int:
        return len(self.visitor_ids)


def build_seen_matrix(train_events: pd.DataFrame, mapping: IdMapping) -> sp.csr_matrix:
    """Sparse ``visitor x item`` mask of interactions already seen in train."""
    rows = train_events["visitor_idx"].to_numpy()
    cols = train_events["item_idx"].to_numpy()
    data = np.ones(len(rows), dtype="bool")
    matrix = sp.csr_matrix(
        (data, (rows, cols)), shape=(mapping.n_visitors, mapping.n_items)
    )
    matrix.sum_duplicates()
    return matrix


def build_evaluation_set(
    events: pd.DataFrame, mapping: IdMapping, split: str, cfg: Config, seen: sp.csr_matrix
) -> EvaluationSet:
    """Collect the target items of every visitor active in ``split``.

    A visitor is evaluated when they have at least one target event in the
    split. Targets are then classified:

    * rankable -- the item exists in ``I_train`` and was not already seen;
    * unrankable because the item never appeared in train;
    * unrankable because it was filtered out as already seen.

    Both unrankable classes stay in ``n_relevant_total``.
    """
    subset = events[
        (events["split"] == split) & events["behavior"].isin(cfg.data.target_behaviors)
    ]
    grouped = subset.groupby("visitorid", sort=True)["itemid"].unique()

    visitor_ids = grouped.index.to_numpy()
    visitor_indices = mapping.visitor_index(pd.Series(visitor_ids)).to_numpy()
    is_warm = visitor_indices >= 0

    relevant: list[np.ndarray] = []
    n_relevant_total: list[int] = []
    n_absent = n_seen_filtered = 0

    for position, targets in enumerate(grouped.to_numpy()):
        item_idx = mapping.item_index(pd.Series(targets)).to_numpy()
        in_train = item_idx >= 0
        n_absent += int((~in_train).sum())

        rankable = item_idx[in_train]
        if cfg.evaluation.filter_seen and is_warm[position] and len(rankable):
            already_seen = np.isin(rankable, seen[visitor_indices[position]].indices)
            n_seen_filtered += int(already_seen.sum())
            rankable = rankable[~already_seen]

        relevant.append(rankable)
        n_relevant_total.append(len(targets))

    evaluation_set = EvaluationSet(
        split=split,
        visitor_ids=visitor_ids,
        visitor_indices=visitor_indices,
        relevant=relevant,
        n_relevant_total=n_relevant_total,
        is_warm=is_warm,
        n_targets_unrankable={
            "item_not_in_train": n_absent,
            "removed_as_already_seen": n_seen_filtered,
            "total_targets": int(sum(n_relevant_total)),
        },
    )
    log.info(
        "eval set %s: %s user (%s warm, %s cold), %s target — %s ngoai I_train, %s bi loc vi da xem",
        split, f"{evaluation_set.n_users:,}", f"{int(is_warm.sum()):,}",
        f"{int((~is_warm).sum()):,}", f"{sum(n_relevant_total):,}",
        f"{n_absent:,}", f"{n_seen_filtered:,}",
    )
    return evaluation_set


class Evaluator:
    """Runs the Top-K protocol for one model on one cohort."""

    def __init__(self, cfg: Config, mapping: IdMapping, seen: sp.csr_matrix) -> None:
        self.cfg = cfg
        self.mapping = mapping
        self.seen = seen
        self.k_values = sorted(cfg.evaluation.k_values)
        self.max_k = max(self.k_values)

    def rank(self, model: Recommender, evaluation_set: EvaluationSet) -> np.ndarray:
        """Produce the top-K item indices for every user of the set.

        Users the model cannot score (cold visitors for a personalised model)
        receive an empty ranking rather than a fabricated one.

        Raises:
            LeakageError: If any recommended item falls outside ``I_train``.
        """
        n_users = evaluation_set.n_users
        rankings = np.full((n_users, self.max_k), -1, dtype="int32")
        scorable = np.arange(n_users)
        if not model.supports_cold_start:
            scorable = scorable[evaluation_set.is_warm]

        batch_size = self.cfg.evaluation.batch_size
        for start in range(0, len(scorable), batch_size):
            positions = scorable[start : start + batch_size]
            # Copy on purpose: a model may legitimately return a broadcast
            # view of its own parameters, and seen-filtering must never write
            # through into the model's state.
            scores = np.array(
                model.score(evaluation_set.visitor_indices[positions]), dtype="float32", copy=True
            )
            if self.cfg.evaluation.filter_seen:
                for offset, position in enumerate(positions):
                    visitor_idx = evaluation_set.visitor_indices[position]
                    if visitor_idx >= 0:
                        scores[offset, self.seen[visitor_idx].indices] = -np.inf
            # Partition on the scores themselves rather than on ``-scores``:
            # negating would duplicate a (batch x n_item) array, and the int64
            # index array argpartition returns is already the dominant cost.
            n_items = scores.shape[1]
            if n_items <= self.max_k:
                ranked = np.argsort(scores, axis=1)[:, ::-1]
            else:
                kth = n_items - self.max_k
                top = np.argpartition(scores, kth, axis=1)[:, kth:]
                order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
                ranked = np.take_along_axis(top, order, axis=1)

            # A filtered item carries -inf and must never surface, however few
            # candidates remain. Its slot becomes -1: an empty slot, not item 0.
            chosen_scores = np.take_along_axis(scores, ranked, axis=1)
            ranked = np.where(np.isfinite(chosen_scores), ranked, -1)
            rankings[positions, : ranked.shape[1]] = ranked[:, : self.max_k]

        # Leakage rule 5, on a real artefact: nothing outside I_train may be
        # recommended. Empty rows (-1) belong to users the model declined.
        recommended = np.unique(rankings[rankings >= 0])
        if len(recommended):
            assert_candidate_scope(
                self.mapping.item_ids[recommended], np.sort(self.mapping.item_ids)
            )
        return rankings

    def evaluate(
        self, model: Recommender, evaluation_set: EvaluationSet
    ) -> tuple[dict[str, object], np.ndarray]:
        """Compute metrics per segment.

        Returns:
            A mapping ``{"warm": {...}, "cold": {...}, "all": {...}}`` plus user
            counts, and the ranking matrix (reused to write ``topk.csv``).
            A segment the model cannot serve is reported as ``None`` -- never as
            a zero that could be mistaken for a measured result.
        """
        rankings = self.rank(model, evaluation_set)
        warm = evaluation_set.is_warm
        masks = {"warm": warm, "cold": ~warm, "all": np.ones_like(warm)}

        results: dict[str, object] = {"n_users": {}, "unrankable_targets": evaluation_set.n_targets_unrankable}
        for segment, mask in masks.items():
            n_users = int(mask.sum())
            results["n_users"][segment] = n_users  # type: ignore[index]

            if segment != "warm" and not model.supports_cold_start:
                # Rule: never mix cold users into a personalised model's metric,
                # and never report a number the model did not actually produce.
                results[segment] = None
                continue
            if n_users == 0:
                results[segment] = None
                continue

            positions = np.flatnonzero(mask)
            results[segment] = compute_metrics(
                top_k_items=[rankings[p] for p in positions],
                relevant_sets=[evaluation_set.relevant[p] for p in positions],
                n_relevant_total=[evaluation_set.n_relevant_total[p] for p in positions],
                k_values=self.k_values,
                n_train_items=self.mapping.n_items,
            )
        return results, rankings

    def top_k_frame(
        self, evaluation_set: EvaluationSet, rankings: np.ndarray, model: Recommender
    ) -> pd.DataFrame:
        """Long-format Top-K table written to ``topk.csv``.

        The demo application replays this file instead of re-scoring, so the
        interface can never show numbers that disagree with the results table
        (luan van muc 3.7.2).
        """
        rows = []
        for position in range(evaluation_set.n_users):
            ranked = rankings[position]
            if (ranked < 0).all():
                continue
            for rank, item_idx in enumerate(ranked[ranked >= 0], start=1):
                rows.append(
                    (
                        int(evaluation_set.visitor_ids[position]),
                        "warm" if evaluation_set.is_warm[position] else "cold",
                        rank,
                        int(self.mapping.item_ids[item_idx]),
                        bool(item_idx in evaluation_set.relevant[position]),
                    )
                )
        return pd.DataFrame(
            rows, columns=["visitor_id", "segment", "rank", "item_id", "is_target"]
        )
