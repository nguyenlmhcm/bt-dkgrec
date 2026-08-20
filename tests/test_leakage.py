"""Tests for the leakage guards (Buoc 3).

Every one of the seven rules in CLAUDE.md has both a passing case and at least
one deliberately corrupted case that MUST raise. Rule 3 gets extra attention:
it is the rule the v11 thesis violated (docs/DECISIONS.md muc D11), so a
regression test injects a single future-dated edge and requires the guard to
catch it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.guards.leakage import (
    LeakageError,
    assert_candidate_scope,
    assert_edges_within_train,
    assert_index_within_mapping,
    assert_latest_record_selected,
    assert_model_selection_scope,
    assert_negatives_in_train,
    assert_side_info_cutoff,
    assert_single_category_per_item,
    assert_temporal_split,
    assert_train_only_mapping,
    run_preprocess_guards,
)

T_TRAIN = 1_000
T_VALID_END = 1_200


@pytest.fixture
def events() -> pd.DataFrame:
    """A clean, correctly split event log."""
    return pd.DataFrame(
        {
            "timestamp": np.array([100, 500, 1000, 1100, 1300, 1400], dtype="int64"),
            "visitorid": np.array([1, 2, 1, 1, 2, 3], dtype="int32"),
            "itemid": np.array([10, 11, 12, 13, 10, 14], dtype="int32"),
            "behavior": pd.Categorical(
                ["view", "view", "addtocart", "view", "transaction", "view"],
                categories=["view", "addtocart", "transaction"],
            ),
            "split": pd.Categorical(
                ["train", "train", "train", "valid", "test", "test"],
                categories=["train", "valid", "test"],
            ),
        }
    )


@pytest.fixture
def side_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Admissible item-category and item-property edges (item_idx in [0, 3))."""
    item_category = pd.DataFrame(
        {
            "item_idx": np.array([0, 1, 2], dtype="int32"),
            "category_id": np.array([100, 100, 200], dtype="int32"),
            "category_idx": np.array([0, 0, 1], dtype="int32"),
            "valid_from": np.array([200, 300, 900], dtype="int64"),
        }
    )
    item_property = pd.DataFrame(
        {
            "item_idx": np.array([0, 0, 1, 2], dtype="int32"),
            "pv_idx": np.array([0, 1, 0, 2], dtype="int32"),
            "valid_from": np.array([150, 400, 800, 1000], dtype="int64"),
        }
    )
    return item_category, item_property


# ══ Rule 1 — split theo thoi gian ═══════════════════════════════════════


def test_rule1_accepts_a_chronological_split(events: pd.DataFrame) -> None:
    assert_temporal_split(events, T_TRAIN, T_VALID_END)


def test_rule1_rejects_a_train_event_after_the_boundary(events: pd.DataFrame) -> None:
    corrupted = events.copy()
    corrupted.loc[0, "timestamp"] = T_TRAIN + 1
    with pytest.raises(LeakageError) as excinfo:
        assert_temporal_split(corrupted, T_TRAIN, T_VALID_END)
    assert excinfo.value.rule == "1"
    assert excinfo.value.kind == "temporal"


def test_rule1_rejects_a_shuffled_random_split(events: pd.DataFrame) -> None:
    """A random split puts old events in test -- exactly what rule 1 forbids."""
    corrupted = events.copy()
    corrupted.loc[0, "split"] = "test"
    with pytest.raises(LeakageError, match="QUY TAC 1"):
        assert_temporal_split(corrupted, T_TRAIN, T_VALID_END)


# ══ Rule 2 — mapping chi tu train ═══════════════════════════════════════


def test_rule2_accepts_a_train_only_mapping(events: pd.DataFrame) -> None:
    assert_train_only_mapping(
        visitor_ids=np.array([1, 2]), item_ids=np.array([10, 11, 12]), events=events
    )


def test_rule2_rejects_an_id_that_leaked_in_from_test(events: pd.DataFrame) -> None:
    with pytest.raises(LeakageError) as excinfo:
        assert_train_only_mapping(
            visitor_ids=np.array([1, 2, 3]),  # visitor 3 only appears in test
            item_ids=np.array([10, 11, 12]),
            events=events,
        )
    assert excinfo.value.rule == "2"
    assert excinfo.value.kind == "identity"
    assert "KHONG co trong train" in str(excinfo.value)


def test_rule2_rejects_an_incomplete_mapping(events: pd.DataFrame) -> None:
    with pytest.raises(LeakageError, match="THIEU trong mapping"):
        assert_train_only_mapping(
            visitor_ids=np.array([1, 2]), item_ids=np.array([10, 11]), events=events
        )


def test_rule2_rejects_raw_ids_in_an_index_column() -> None:
    """The bug found in Buoc 2: item_idx held raw itemid values."""
    frame = pd.DataFrame({"item_idx": np.array([0, 1, 466_864], dtype="int64")})
    with pytest.raises(LeakageError, match="ngoai khong gian chi so"):
        assert_index_within_mapping(frame, "item_idx", n_entities=3, name="item_property")


def test_rule2_accepts_indices_inside_the_mapping() -> None:
    frame = pd.DataFrame({"item_idx": np.array([0, 1, 2], dtype="int32")})
    assert_index_within_mapping(frame, "item_idx", n_entities=3, name="item_property")


# ══ Rule 3 — side info <= T_train  (rule uu tien) ═══════════════════════


def test_rule3_accepts_side_info_within_the_cutoff(side_tables) -> None:
    item_category, item_property = side_tables
    assert_side_info_cutoff(item_category, T_TRAIN, "item_category")
    assert_side_info_cutoff(item_property, T_TRAIN, "item_property")


def test_rule3_regression_single_injected_future_edge_must_fail(side_tables) -> None:
    """Regression for the v11 defect: ONE future-dated edge must be caught.

    v11 loaded item attributes from every timestamp, so its training graph saw
    the future state of products. Rebuilding with the cutoff removed reproduced
    its edge count to +1.1%, which is what identified the defect.
    """
    _, item_property = side_tables
    leaked = pd.concat(
        [
            item_property,
            pd.DataFrame(
                {
                    "item_idx": np.array([1], dtype="int32"),
                    "pv_idx": np.array([3], dtype="int32"),
                    "valid_from": np.array([T_TRAIN + 1], dtype="int64"),
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(LeakageError) as excinfo:
        assert_side_info_cutoff(leaked, T_TRAIN, "item_property")

    error = excinfo.value
    assert error.rule == "3"
    assert error.kind == "temporal"
    assert error.n_violations == 1
    message = str(error)
    assert "1 / 5" in message                 # dem duoc so dong vi pham
    assert str(T_TRAIN) in message            # in ra moc thoi gian
    assert "Vi du" in message                 # co dong vi du de chan doan
    assert "KHONG duoc tat guard" in message


def test_rule3_rejects_a_table_without_a_timestamp_column() -> None:
    """A table with no valid_from cannot be verified, so it is not trusted."""
    frame = pd.DataFrame({"item_idx": [0, 1], "pv_idx": [0, 1]})
    with pytest.raises(LeakageError, match="khong co cot"):
        assert_side_info_cutoff(frame, T_TRAIN, "item_property")


def test_rule3_boundary_timestamp_is_admissible(side_tables) -> None:
    """valid_from == T_train is inside the training window, not a violation."""
    _, item_property = side_tables
    assert int(item_property["valid_from"].max()) == T_TRAIN
    assert_side_info_cutoff(item_property, T_TRAIN, "item_property")


def test_rule3_rejects_interaction_edges_built_after_the_cutoff(events: pd.DataFrame) -> None:
    with pytest.raises(LeakageError, match="canh tuong tac"):
        assert_edges_within_train(events, T_TRAIN)


def test_rule3_accepts_interaction_edges_from_train_only(events: pd.DataFrame) -> None:
    assert_edges_within_train(events[events["split"] == "train"], T_TRAIN)


# ══ Rule 4 — ban ghi moi nhat <= T_train ════════════════════════════════


def test_rule4_accepts_one_category_per_item(side_tables) -> None:
    item_category, _ = side_tables
    assert_single_category_per_item(item_category)


def test_rule4_rejects_two_categories_for_one_item(side_tables) -> None:
    item_category, _ = side_tables
    corrupted = pd.concat([item_category, item_category.iloc[[0]]], ignore_index=True)
    with pytest.raises(LeakageError) as excinfo:
        assert_single_category_per_item(corrupted)
    assert excinfo.value.rule == "4"


def test_rule4_accepts_the_newest_record_per_group() -> None:
    items = np.array([7, 7, 7, 8])
    props = np.array([1, 1, 1, 1])
    stamps = np.array([100, 300, 200, 50])
    assert_latest_record_selected([items, props], stamps, selected=np.array([1, 3]))


def test_rule4_rejects_taking_the_last_line_of_the_file() -> None:
    """The classic trap: position 2 is the file's last row but not the newest."""
    items = np.array([7, 7, 7, 8])
    props = np.array([1, 1, 1, 1])
    stamps = np.array([100, 300, 200, 50])
    with pytest.raises(LeakageError, match="KHONG phai ban ghi moi nhat"):
        assert_latest_record_selected([items, props], stamps, selected=np.array([2, 3]))


def test_rule4_rejects_keeping_two_snapshots_of_one_group() -> None:
    items = np.array([7, 7])
    props = np.array([1, 1])
    stamps = np.array([100, 100])
    with pytest.raises(LeakageError, match="khac so nhom"):
        assert_latest_record_selected([items, props], stamps, selected=np.array([0, 1]))


# ══ Rule 5 — candidate ⊆ I_train ════════════════════════════════════════


def test_rule5_accepts_candidates_inside_the_train_universe() -> None:
    assert_candidate_scope(np.array([10, 11]), np.array([10, 11, 12]))


def test_rule5_rejects_a_future_item_in_the_candidate_set() -> None:
    with pytest.raises(LeakageError) as excinfo:
        assert_candidate_scope(np.array([10, 99]), np.array([10, 11, 12]))
    assert excinfo.value.rule == "5"
    assert "99" in str(excinfo.value)


# ══ Rule 6 — negative sampling chi tu I_train ═══════════════════════════


def test_rule6_accepts_negatives_drawn_from_train_items() -> None:
    assert_negatives_in_train(np.array([10, 12, 10]), np.array([10, 11, 12]))


def test_rule6_rejects_a_negative_outside_the_train_universe() -> None:
    with pytest.raises(LeakageError) as excinfo:
        assert_negatives_in_train(np.array([10, 77]), np.array([10, 11, 12]))
    assert excinfo.value.rule == "6"


# ══ Rule 7 — model selection chi doc valid ══════════════════════════════


def test_rule7_accepts_selection_on_validation() -> None:
    assert_model_selection_scope("ndcg@20", consulted_splits=["valid"])


def test_rule7_rejects_a_test_metric_as_monitor() -> None:
    with pytest.raises(LeakageError) as excinfo:
        assert_model_selection_scope("test_ndcg@20", consulted_splits=["valid"])
    assert excinfo.value.rule == "7"


def test_rule7_rejects_consulting_test_metrics_at_all() -> None:
    with pytest.raises(LeakageError, match="doc metric cua tap test"):
        assert_model_selection_scope("ndcg@20", consulted_splits=["valid", "test"])


# ══ Gate tong hop ═══════════════════════════════════════════════════════


def test_gate_passes_on_a_clean_dataset(events: pd.DataFrame, side_tables) -> None:
    item_category, item_property = side_tables
    passed = run_preprocess_guards(
        events=events,
        visitor_ids=np.array([1, 2]),
        item_ids=np.array([10, 11, 12]),
        item_category=item_category,
        item_property=item_property,
        t_train=T_TRAIN,
        t_valid_end=T_VALID_END,
        monitor="ndcg@20",
    )
    assert len(passed) == 7
    # Rules 5 and 6 are deliberately absent: nothing to check before an
    # evaluator or a sampler exists, and a vacuous PASS would mislead.
    assert not any("rule 5" in name or "rule 6" in name for name in passed)


def test_gate_fails_on_the_v11_defect(events: pd.DataFrame, side_tables) -> None:
    """End-to-end: one future-dated property edge stops the whole preprocess."""
    item_category, item_property = side_tables
    item_property = item_property.copy()
    item_property.loc[0, "valid_from"] = T_TRAIN + 1
    with pytest.raises(LeakageError, match="QUY TAC 3"):
        run_preprocess_guards(
            events=events,
            visitor_ids=np.array([1, 2]),
            item_ids=np.array([10, 11, 12]),
            item_category=item_category,
            item_property=item_property,
            t_train=T_TRAIN,
            t_valid_end=T_VALID_END,
            monitor="ndcg@20",
        )
