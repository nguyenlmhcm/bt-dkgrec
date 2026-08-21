"""Ranking metrics -- the single definition used by every model.

No model computes its own metric (CLAUDE.md, DRY rule 3). Everything reported in
the thesis flows through these four functions.

Conservative candidate policy
-----------------------------
A target item that never appeared in train cannot be ranked, because the
candidate set is ``I_train``. Such a target counts as a **miss** and stays in
the denominator; it is never dropped from the sample. That makes every number
here a lower bound rather than a flattered one (CLAUDE.md muc "Chinh sach
candidate bao thu").

Definitions
-----------
For a user ``u`` with target set ``R_u`` (all targets, rankable or not) and a
ranked list ``L_u`` of length K drawn from ``I_train``::

    Recall@K   = |L_u ∩ R_u| / |R_u|
    HitRate@K  = 1 if |L_u ∩ R_u| > 0 else 0
    NDCG@K     = DCG@K / IDCG@K,  DCG@K = sum over hits of 1/log2(rank+2)
                 IDCG@K = sum over the first min(|R_u|, K) positions
    Coverage@K = |distinct items appearing in any L_u| / |I_train|

Coverage is a corpus-level metric: its numerator counts distinct items across
**all** evaluated users, and its denominator is the size of the train item
universe (docs/DECISIONS.md muc D5).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Metric names produced by :func:`compute_metrics`, in report order.
METRIC_NAMES = ("recall", "ndcg", "hit_rate", "coverage")


def _discounts(k: int) -> np.ndarray:
    """Positional discounts ``1 / log2(rank + 2)`` for ranks ``0..k-1``."""
    return 1.0 / np.log2(np.arange(k, dtype="float64") + 2.0)


def recall_at_k(hits: np.ndarray, n_relevant: int, k: int) -> float:
    """Fraction of a user's targets that appear in the top ``k``.

    Args:
        hits: Boolean array over the ranked list; ``hits[i]`` is True when the
            item at rank ``i`` is a target.
        n_relevant: Size of the user's full target set, including targets that
            could not be ranked at all.
        k: Cut-off.
    """
    if n_relevant <= 0:
        return 0.0
    return float(hits[:k].sum()) / float(n_relevant)


def hit_rate_at_k(hits: np.ndarray, k: int) -> float:
    """1.0 when at least one target is retrieved in the top ``k``."""
    return 1.0 if bool(hits[:k].any()) else 0.0


def ndcg_at_k(hits: np.ndarray, n_relevant: int, k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    The ideal ranking places ``min(n_relevant, k)`` targets at the top. Because
    ``n_relevant`` counts unrankable targets too, a user whose targets are all
    absent from ``I_train`` scores 0 -- consistent with the conservative policy.
    """
    if n_relevant <= 0:
        return 0.0
    discounts = _discounts(k)
    dcg = float((hits[:k].astype("float64") * discounts).sum())
    ideal = float(discounts[: min(n_relevant, k)].sum())
    return dcg / ideal if ideal > 0 else 0.0


def coverage_at_k(top_k_items: Sequence[np.ndarray], n_train_items: int, k: int) -> float:
    """Share of the train catalogue that the model actually recommends.

    Args:
        top_k_items: One ranked item-index array per evaluated user.
        n_train_items: ``|I_train|`` -- the denominator.
        k: Cut-off.
    """
    if n_train_items <= 0 or not len(top_k_items):
        return 0.0
    recommended = np.concatenate([row[:k] for row in top_k_items])
    # -1 marks an empty slot (fewer candidates than K, or a user the model
    # declined to score). It is not an item and must not inflate coverage.
    distinct = np.unique(recommended[recommended >= 0])
    return float(len(distinct)) / float(n_train_items)


def compute_metrics(
    top_k_items: Sequence[np.ndarray],
    relevant_sets: Sequence[np.ndarray],
    n_relevant_total: Sequence[int],
    k_values: Sequence[int],
    n_train_items: int,
) -> dict[str, float]:
    """Average every metric over a set of users.

    Args:
        top_k_items: Ranked item indices per user, longest cut-off first.
        relevant_sets: Rankable target item indices per user.
        n_relevant_total: Full target count per user, including unrankable
            targets -- this is what keeps the evaluation conservative.
        k_values: Cut-offs to report, e.g. ``[10, 20]``.
        n_train_items: ``|I_train|``.

    Returns:
        Mapping like ``{"recall@20": 0.0216, ...}``. Empty user set yields 0.0
        for every metric rather than a division error.

    Raises:
        ValueError: If the per-user sequences have different lengths.
    """
    n_users = len(top_k_items)
    if not (n_users == len(relevant_sets) == len(n_relevant_total)):
        raise ValueError(
            "top_k_items, relevant_sets va n_relevant_total phai cung do dai; "
            f"dang co {n_users}, {len(relevant_sets)}, {len(n_relevant_total)}"
        )

    results: dict[str, float] = {}
    if n_users == 0:
        for k in k_values:
            for metric in METRIC_NAMES:
                results[f"{metric}@{k}"] = 0.0
        return results

    hit_masks = [np.isin(ranked, relevant) for ranked, relevant in zip(top_k_items, relevant_sets, strict=True)]
    for k in k_values:
        recalls, ndcgs, hit_rates = [], [], []
        for hits, n_relevant in zip(hit_masks, n_relevant_total, strict=True):
            recalls.append(recall_at_k(hits, n_relevant, k))
            ndcgs.append(ndcg_at_k(hits, n_relevant, k))
            hit_rates.append(hit_rate_at_k(hits, k))
        results[f"recall@{k}"] = float(np.mean(recalls))
        results[f"ndcg@{k}"] = float(np.mean(ndcgs))
        results[f"hit_rate@{k}"] = float(np.mean(hit_rates))
        results[f"coverage@{k}"] = coverage_at_k(top_k_items, n_train_items, k)
    return results
