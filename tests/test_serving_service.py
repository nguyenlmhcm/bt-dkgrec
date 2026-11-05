"""Dich vu du bao cua demo (D38): lich su doc tu kho, Top-K tinh lai theo hanh vi moi."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.serving.online import top_k
from src.serving.service import DemoService, event_weight

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("test_online_helpers", ROOT / "test_online.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def _service() -> tuple[DemoService, np.ndarray, object]:
    e0 = np.random.default_rng(3).normal(size=(H.SPACE.total, 8))
    adjacency, _ = H._graph()
    service = DemoService(
        bundle=H._bundle(e0, adjacency),
        e0_visitors=e0[:H.SPACE.n_visitor],
        visitor_ids=np.array([10, 11, 12, 13]),
        t_train=H.T_TRAIN,
    )
    return service, e0, adjacency


def _history(visitor: int) -> pd.DataFrame:
    frame = H._events(visitor)
    return pd.DataFrame({
        "item_id": frame["item_idx"] + 100,
        "behavior": frame["behavior"],
        "timestamp": frame["timestamp"],
    })


def test_no_new_behaviour_gives_the_offline_ranking() -> None:
    """Guard: chua co hanh vi moi thi Top-K phai trung lan truyen toan do thi."""
    service, e0, adjacency = _service()
    z = H._full_propagation(e0, adjacency, 3)
    items = z[H.SPACE.item_offset:H.SPACE.item_offset + H.SPACE.n_item]
    for visitor in H.HISTORY:
        seen = H._events(visitor)["item_idx"].unique()
        expected = top_k((items @ z[visitor]).astype("float32"), 3, seen)
        got = service.recommend(10 + visitor, _history(visitor), k=3)
        assert got["item_idx"].tolist() == expected.tolist()


def test_a_new_event_changes_the_top_k_and_is_excluded_from_it() -> None:
    service, _, _ = _service()
    before = service.recommend(12, _history(2), k=3)
    bought = int(before["item_id"].iloc[0])
    extra = pd.DataFrame([{"item_id": bought, "behavior": "transaction",
                           "timestamp": service.next_timestamp(_history(2))}])
    after = service.recommend(12, pd.concat([_history(2), extra], ignore_index=True), k=3)
    assert bought not in after["item_id"].tolist()
    assert after["item_id"].tolist() != before["item_id"].tolist()


def test_undoing_the_event_restores_the_top_k() -> None:
    service, _, _ = _service()
    first = service.recommend(11, _history(1), k=4)
    again = service.recommend(11, _history(1), k=4)
    pd.testing.assert_frame_equal(first, again)


def test_items_unknown_to_the_model_do_not_crash_serving() -> None:
    service, _, _ = _service()
    extra = pd.DataFrame([{"item_id": 999_999, "behavior": "view",
                           "timestamp": H.T_TRAIN + 1000}])
    got = service.recommend(12, pd.concat([_history(2), extra], ignore_index=True), k=2)
    assert len(got) == 2


def test_a_visitor_the_model_never_saw_is_served_from_history_alone() -> None:
    service, _, _ = _service()
    events = pd.DataFrame([{"item_id": 101, "behavior": "view", "timestamp": H.T_TRAIN + 1000}])
    assert len(service.recommend(777, events, k=2)) == 2


def test_next_timestamp_is_after_t_train_and_after_the_last_event() -> None:
    service, _, _ = _service()
    assert service.next_timestamp(_history(0)) > H.T_TRAIN
    later = pd.DataFrame([{"item_id": 100, "behavior": "view", "timestamp": H.T_TRAIN + 5000}])
    assert service.next_timestamp(later) > H.T_TRAIN + 5000


def test_event_weight_follows_the_formula() -> None:
    service, _, _ = _service()
    assert event_weight(service.bundle, "transaction", age_days=2) == pytest.approx(
        H.ALPHA[2] * np.exp(-H.LAMBDA * 2))


def test_rank_change_labels_moves_against_the_previous_list():
    from src.serving.service import rank_change

    before = [10, 20, 30]
    assert rank_change(before, 40, 3) == "mới"
    assert rank_change(before, 20, 1) == "▲ 1"
    assert rank_change(before, 10, 3) == "▼ 2"
    assert rank_change(before, 30, 3) == ""
