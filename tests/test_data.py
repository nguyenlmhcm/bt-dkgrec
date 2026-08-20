"""Tests for the data layer (Buoc 2).

Hermetic: every test builds a tiny synthetic dataset, so they run in
milliseconds and do not depend on the 1.3 GB raw files. The rules being locked
in are the leakage rules of CLAUDE.md, above all rule 4 -- newest admissible
record wins, never the last line of the file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cohort import apply_cohort, select_cohort_visitors
from src.data.mapping import IdMapping
from src.data.side_info import _latest_per_group
from src.data.splitter import TemporalSplit, split_events
from src.utils.config import load_config


@pytest.fixture
def events() -> pd.DataFrame:
    """Ten events on a clean 0..900 timeline, three visitors."""
    return pd.DataFrame(
        {
            "timestamp": np.arange(0, 1000, 100, dtype="int64"),
            "visitorid": np.array([1, 1, 1, 1, 1, 1, 2, 2, 3, 3], dtype="int32"),
            "itemid": np.array([10, 11, 12, 13, 14, 15, 20, 21, 30, 31], dtype="int32"),
            "behavior": pd.Categorical(
                ["view"] * 6 + ["addtocart", "view", "view", "transaction"],
                categories=["view", "addtocart", "transaction"],
            ),
        }
    )


# ── Split ────────────────────────────────────────────────────────────────


def test_split_cuts_on_time_span_not_event_count(events: pd.DataFrame) -> None:
    split = load_config(model="popularity").data.split
    labelled, boundaries = split_events(events, split)

    # span = 900 -> T_train = 0 + int(0.7*900) = 630, T_valid_end = 720
    assert boundaries.t_min == 0
    assert boundaries.t_max == 900
    assert boundaries.t_train == 630
    assert boundaries.t_valid_end == 720
    assert list(labelled["split"]) == ["train"] * 7 + ["valid"] + ["test"] * 2


def test_split_boundary_is_inclusive() -> None:
    boundaries = TemporalSplit(t_min=0, t_max=100, t_train=70, t_valid_end=80, boundary_inclusive=True)
    labels = boundaries.assign(pd.Series([69, 70, 71, 80, 81]))
    assert list(labels) == ["train", "train", "valid", "valid", "test"]


def test_split_never_shuffles_time_order(events: pd.DataFrame) -> None:
    """A train event can never be newer than a test event (leakage rule 1)."""
    split = load_config(model="popularity").data.split
    labelled, _ = split_events(events, split)
    newest_train = labelled.loc[labelled["split"] == "train", "timestamp"].max()
    oldest_test = labelled.loc[labelled["split"] == "test", "timestamp"].min()
    assert newest_train < oldest_test


# ── Mapping ──────────────────────────────────────────────────────────────


def test_mapping_is_built_from_train_only(events: pd.DataFrame) -> None:
    split = load_config(model="popularity").data.split
    labelled, _ = split_events(events, split)
    mapping = IdMapping.from_train_events(labelled[labelled["split"] == "train"].drop(columns="split"))

    assert mapping.n_items == 7  # items 10..15 and 20 -- not the test-only items
    assert 31 not in set(mapping.item_ids)


def test_mapping_rejects_non_train_rows(events: pd.DataFrame) -> None:
    split = load_config(model="popularity").data.split
    labelled, _ = split_events(events, split)
    with pytest.raises(ValueError, match="train rows only"):
        IdMapping.from_train_events(labelled)


def test_unknown_ids_map_to_minus_one() -> None:
    mapping = IdMapping(
        visitor_ids=np.array([5, 9, 12], dtype="int32"),
        item_ids=np.array([100, 200], dtype="int32"),
    )
    assert list(mapping.visitor_index(pd.Series([5, 7, 12, 99]))) == [0, -1, 2, -1]
    assert list(mapping.item_index(pd.Series([100, 150, 200]))) == [0, -1, 1]


def test_mapping_roundtrips_through_parquet(tmp_path) -> None:
    mapping = IdMapping(
        visitor_ids=np.array([3, 8, 11], dtype="int32"),
        item_ids=np.array([1, 4], dtype="int32"),
    )
    mapping.save(tmp_path)
    restored = IdMapping.load(tmp_path)
    assert np.array_equal(restored.visitor_ids, mapping.visitor_ids)
    assert np.array_equal(restored.item_ids, mapping.item_ids)


# ── Cohort ───────────────────────────────────────────────────────────────


def test_active_cohort_thresholds_on_train_events(events: pd.DataFrame) -> None:
    split = load_config(model="popularity").data.split
    labelled, _ = split_events(events, split)
    train = labelled[labelled["split"] == "train"]

    original = load_config(model="popularity", cohort="original").cohort
    active = load_config(model="popularity", cohort="active").cohort

    assert list(select_cohort_visitors(train, original)) == [1, 2]
    assert list(select_cohort_visitors(train, active)) == [1]  # only visitor 1 has >= 5


def test_cohort_filter_applies_to_every_split(events: pd.DataFrame) -> None:
    split = load_config(model="popularity").data.split
    labelled, _ = split_events(events, split)
    filtered = apply_cohort(labelled, np.array([1], dtype="int32"))
    assert set(filtered["visitorid"]) == {1}
    assert set(filtered["split"]) <= {"train", "valid", "test"}


# ── Leakage rule 4: newest admissible record wins ────────────────────────


def test_latest_per_group_picks_newest_not_last_line() -> None:
    """The trap: the last line of the file is older than an earlier line."""
    items = np.array([7, 7, 7, 8, 8])
    props = np.array([1, 1, 1, 1, 1])
    stamps = np.array([100, 300, 200, 50, 40])  # newest for item 7 is at position 1
    picked = _latest_per_group([items, props], stamps)
    assert sorted(picked.tolist()) == [1, 3]
    assert stamps[picked].tolist() in ([300, 50], [50, 300])


def test_latest_per_group_separates_properties_of_one_item() -> None:
    items = np.array([7, 7, 7, 7])
    props = np.array([1, 1, 2, 2])
    stamps = np.array([10, 20, 30, 15])
    picked = np.sort(_latest_per_group([items, props], stamps))
    assert picked.tolist() == [1, 2]  # newest of property 1, newest of property 2


def test_latest_per_group_is_deterministic_on_ties() -> None:
    items = np.array([1, 1])
    props = np.array([0, 0])
    stamps = np.array([5, 5])
    first = _latest_per_group([items, props], stamps)
    second = _latest_per_group([items, props], stamps)
    assert first.tolist() == second.tolist()
    assert len(first) == 1


def test_latest_per_group_handles_single_row() -> None:
    picked = _latest_per_group([np.array([1]), np.array([2])], np.array([9]))
    assert picked.tolist() == [0]
