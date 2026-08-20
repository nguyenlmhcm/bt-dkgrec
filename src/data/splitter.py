"""Temporal train/valid/test split.

The split cuts on quantiles of the *time range*, never on event counts and never
at random (leakage rule 1). Verified against the v11 thesis: with
``T_train = t_min + int(0.7 * (t_max - t_min))`` and an inclusive boundary,
RetailRocket yields exactly 2,024,042 train events (docs/DECISIONS.md muc D1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.config import SplitConfig
from src.utils.logging import get_logger

log = get_logger(__name__)

SPLIT_LABELS = ("train", "valid", "test")


@dataclass(frozen=True)
class TemporalSplit:
    """Absolute timestamp boundaries of the temporal split (milliseconds)."""

    t_min: int
    t_max: int
    t_train: int
    t_valid_end: int
    boundary_inclusive: bool

    @classmethod
    def from_timestamps(cls, timestamps: pd.Series, split: SplitConfig) -> "TemporalSplit":
        """Derive boundaries from the full event log.

        Args:
            timestamps: Every event timestamp in the dataset, in milliseconds.
            split: Validated split configuration.
        """
        if split.mode != "time_span":
            raise ValueError(f"unsupported split mode: {split.mode}")
        t_min, t_max = int(timestamps.min()), int(timestamps.max())
        span = t_max - t_min
        # Round the cumulative ratio before scaling: in IEEE floats
        # 0.7 + 0.1 == 0.7999999999999999, which would move the validation
        # boundary by one millisecond and silently reassign events.
        cumulative_valid = round(split.train_ratio + split.valid_ratio, 10)
        boundaries = cls(
            t_min=t_min,
            t_max=t_max,
            t_train=t_min + int(round(split.train_ratio, 10) * span),
            t_valid_end=t_min + int(cumulative_valid * span),
            boundary_inclusive=split.boundary_inclusive,
        )
        log.info(
            "temporal split: span=%.1f days, T_train=%d, T_valid_end=%d",
            span / 86_400_000, boundaries.t_train, boundaries.t_valid_end,
        )
        return boundaries

    def assign(self, timestamps: pd.Series) -> pd.Categorical:
        """Label each timestamp ``train`` / ``valid`` / ``test``."""
        ts = timestamps.to_numpy()
        in_train = ts <= self.t_train if self.boundary_inclusive else ts < self.t_train
        in_valid = (~in_train) & (ts <= self.t_valid_end)
        labels = np.where(in_train, "train", np.where(in_valid, "valid", "test"))
        return pd.Categorical(labels, categories=SPLIT_LABELS)

    def as_dict(self) -> dict[str, int | bool]:
        """Serialisable form written to ``split.json``."""
        return {
            "t_min": self.t_min,
            "t_max": self.t_max,
            "t_train": self.t_train,
            "t_valid_end": self.t_valid_end,
            "boundary_inclusive": self.boundary_inclusive,
            "span_days": round((self.t_max - self.t_min) / 86_400_000, 3),
        }


def split_events(events: pd.DataFrame, split: SplitConfig) -> tuple[pd.DataFrame, TemporalSplit]:
    """Attach a ``split`` column to the event log.

    Returns:
        The events with a categorical ``split`` column, and the boundaries used.
    """
    boundaries = TemporalSplit.from_timestamps(events["timestamp"], split)
    out = events.copy()
    out["split"] = boundaries.assign(out["timestamp"])
    sizes = out["split"].value_counts()
    log.info(
        "split sizes: train=%s valid=%s test=%s",
        f"{sizes.get('train', 0):,}", f"{sizes.get('valid', 0):,}", f"{sizes.get('test', 0):,}",
    )
    return out, boundaries
