"""Raw RetailRocket readers.

All readers declare explicit dtypes and read only the columns that are used.
``item_properties`` (20.3M rows / 852 MB across two files) is never loaded whole:
it is streamed in chunks and filtered inside each chunk, because preprocessing
runs on a 3 GB VPS. See docs/DECISIONS.md muc D9.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)

#: Column names produced by :func:`load_events`; ``event`` is renamed to ``behavior``.
EVENT_COLUMNS = ("timestamp", "visitorid", "itemid", "behavior")

_EVENT_DTYPES = {"timestamp": "int64", "visitorid": "int32", "itemid": "int32", "event": "category"}
_PROPERTY_DTYPES = {"timestamp": "int64", "itemid": "int32", "property": "str", "value": "str"}


def raw_dir(cfg: Config) -> Path:
    """Absolute path of the raw data directory."""
    return cfg.paths.resolved()["raw"]


def load_events(cfg: Config) -> pd.DataFrame:
    """Load the full event log.

    Returns:
        DataFrame with columns :data:`EVENT_COLUMNS`, sorted by timestamp.
        ``transactionid`` is dropped: the task is Top-K ranking of target
        behaviors, not transaction reconstruction.

    Raises:
        ValueError: If an unexpected behavior label is present.
    """
    path = raw_dir(cfg) / cfg.data.events_file
    df = pd.read_csv(path, usecols=list(_EVENT_DTYPES), dtype=_EVENT_DTYPES)
    df = df.rename(columns={"event": "behavior"})

    unexpected = set(df["behavior"].cat.categories) - set(cfg.data.history_behaviors)
    if unexpected:
        raise ValueError(f"unexpected behavior labels in {path.name}: {sorted(unexpected)}")

    df = df.sort_values("timestamp", kind="mergesort", ignore_index=True)
    log.info("loaded %s: %s rows", path.name, f"{len(df):,}")
    return df


def load_category_tree(cfg: Config) -> pd.DataFrame:
    """Load the category hierarchy.

    Returns:
        DataFrame ``[categoryid int32, parentid Int32]``. ``parentid`` is a
        nullable integer: root categories have no parent.
    """
    path = raw_dir(cfg) / cfg.data.category_tree_file
    df = pd.read_csv(path, dtype={"categoryid": "int32", "parentid": "Int32"})
    log.info("loaded %s: %s categories, %s roots",
             path.name, f"{len(df):,}", f"{int(df['parentid'].isna().sum()):,}")
    return df


def iter_item_properties(cfg: Config) -> Iterator[pd.DataFrame]:
    """Stream both ``item_properties`` files in chunks.

    Yields:
        Chunks with columns ``[timestamp, itemid, property, value]``. Chunk size
        comes from ``cfg.data.chunksize``; nothing is accumulated here, so peak
        memory stays proportional to one chunk.
    """
    for filename in cfg.data.item_properties_files:
        path = raw_dir(cfg) / filename
        log.info("streaming %s (chunksize=%s)", path.name, f"{cfg.data.chunksize:,}")
        for chunk in pd.read_csv(path, dtype=_PROPERTY_DTYPES, chunksize=cfg.data.chunksize):
            yield chunk


def audit_events(events: pd.DataFrame, cfg: Config) -> dict[str, int]:
    """Compute the raw-data audit figures printed by ``01_preprocess.py``.

    These are sanity checks against CLAUDE.md muc "Dac ta du lieu RetailRocket",
    not targets to be matched by tuning.
    """
    counts = events["behavior"].value_counts()
    audit = {
        "total_events": len(events),
        "unique_visitors": int(events["visitorid"].nunique()),
        "unique_items": int(events["itemid"].nunique()),
        "t_min": int(events["timestamp"].min()),
        "t_max": int(events["timestamp"].max()),
    }
    for behavior in cfg.data.history_behaviors:
        audit[f"n_{behavior}"] = int(counts.get(behavior, 0))
    audit["n_target_events"] = int(events["behavior"].isin(cfg.data.target_behaviors).sum())
    return audit
