"""Schema of ``curves.csv`` -- the per-epoch evidence of how training went.

Kept out of :mod:`src.training.trainer` on purpose: that module imports torch,
while ``scripts/03_train.py`` must stay importable on a machine without it. The
heuristic baselines run on the torch-free VPS (docs/DECISIONS.md muc D28) and
write a curves file of the very same shape, so the shape itself has to live
somewhere both can reach.

One row per epoch
-----------------
The loss is recorded on **every** epoch and the validation block on evaluation
epochs (``evaluated`` says which). Recording only evaluation epochs, as the
first version did, threw away four fifths of the loss curve -- the single most
convincing piece of convergence evidence -- for no saving at all, since the
numbers were already computed and held in memory.

Every metric, not only the monitored one
----------------------------------------
``Evaluator.evaluate`` computes the whole warm block on each call. Keeping one
number and dropping seven meant that a reader could not check whether the other
metrics had also flattened when patience ran out -- the standard objection to an
early-stopped baseline. They cost nothing to keep.
"""

from __future__ import annotations

import pandas as pd

#: Columns identifying which run a row belongs to. Written by the training
#: script rather than the trainer, which is unaware of cohorts and seeds. They
#: make a concatenation of many ``curves.csv`` files self-describing: a run id
#: like ``original_recent_popularity_2020_...`` cannot be split back into cohort
#: and model by string surgery, because model names contain underscores too.
IDENTITY_COLUMNS = ("model", "cohort", "seed")

#: Per-epoch columns that always exist, in order, before the metric block.
FIXED_HEAD = ("epoch", "loss", "seconds", "evaluated")

#: Columns that always come last.
FIXED_TAIL = ("note",)

#: Prefix marking a validation-metric column. The split is in the name so that
#: no reader can mistake these for test numbers (leakage rule 7).
VALID_PREFIX = "valid_"


def valid_column(metric: str) -> str:
    """Column name holding ``metric`` measured on the validation split."""
    return f"{VALID_PREFIX}{metric}"


def order_columns(frame: pd.DataFrame, monitor: str) -> pd.DataFrame:
    """Return ``frame`` with the canonical column order and no missing column.

    The monitored metric's column is guaranteed to exist even when no
    evaluation ever produced a number, so that a run without validation still
    concatenates with runs that had it.

    Args:
        frame: Rows collected during training, or the single heuristic row.
        monitor: ``cfg.training.monitor`` -- the metric that drove selection.
    """
    frame = frame.copy()
    metric_columns = sorted(
        {*(c for c in frame.columns if c.startswith(VALID_PREFIX)), valid_column(monitor)}
    )
    ordered = [*FIXED_HEAD, *metric_columns, *FIXED_TAIL]
    for column in ordered:
        if column not in frame.columns:
            frame[column] = None
    return frame[ordered]


def add_identity(frame: pd.DataFrame, model: str, cohort: str, seed: int) -> pd.DataFrame:
    """Prefix the identity columns so the file stands alone outside its folder."""
    frame = frame.copy()
    for column, value in zip(IDENTITY_COLUMNS, (model, cohort, seed)):
        frame[column] = value
    return frame[[*IDENTITY_COLUMNS, *(c for c in frame.columns if c not in IDENTITY_COLUMNS)]]
