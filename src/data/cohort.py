"""Cohort selection.

Two cohorts are evaluated:

``original``
    Every visitor. This is the **main** evaluation set of the thesis.
``active``
    Visitors with ``|E_u^train| >= min_active_events``. The threshold was fixed
    *before* mappings and graphs were rebuilt, so results on this cohort are an
    exploratory subgroup analysis and never replace the main set.

The filter is applied to **train** events only; validation and test are then
restricted to the surviving visitors so the cohort is consistent end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import CohortConfig
from src.utils.logging import get_logger

log = get_logger(__name__)


def select_cohort_visitors(train_events: pd.DataFrame, cohort: CohortConfig) -> np.ndarray:
    """Return the sorted visitor ids belonging to the cohort.

    Args:
        train_events: Train-split events only. Passing anything else would let
            future activity decide who is evaluated (leakage).
        cohort: Validated cohort configuration.
    """
    counts = train_events["visitorid"].value_counts()
    if cohort.min_active_events > 0:
        counts = counts[counts >= cohort.min_active_events]
    visitors = np.sort(counts.index.to_numpy())
    log.info(
        "cohort %r: %s visitors (nguong |E_u^train| >= %d)",
        cohort.name, f"{len(visitors):,}", cohort.min_active_events,
    )
    return visitors


def apply_cohort(events: pd.DataFrame, visitors: np.ndarray) -> pd.DataFrame:
    """Keep only events of the cohort visitors, across every split."""
    mask = events["visitorid"].isin(visitors)
    out = events.loc[mask].reset_index(drop=True)
    log.info("cohort filter kept %s / %s events", f"{len(out):,}", f"{len(events):,}")
    return out
