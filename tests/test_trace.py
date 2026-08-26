"""Tests cho lop truy vet cua demo (Buoc 10).

Rang buoc quan trong nhat khong phai giao dien dep, ma la: **demo khong duoc
cham diem lai**. No phat lai `topk.csv` cua run artifact, nen so tren man hinh
luon trung bang ket qua trong de an (luan van muc 3.7.2). Mot test o day khang
dinh dieu do o muc ma nguoi ta khong the vo tinh pha.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.export.trace import (
    TraceUnavailableError,
    explain_recommendation,
    load_topk,
    load_trace,
    visitor_edges,
    visitor_history,
    visitor_recommendations,
)

DAY = 86_400_000


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    """Mot ban xuat toi thieu, dung schema ma 07_export_neo4j.py sinh ra."""
    out = tmp_path / "export"
    out.mkdir()
    pd.DataFrame([
        ("1_10_100", 1, 10, "view", 90 * DAY, 0.9),
        ("1_10_200", 1, 10, "addtocart", 95 * DAY, 1.8),
        ("1_11_300", 1, 11, "view", 99 * DAY, 0.95),
        ("2_10_400", 2, 10, "view", 50 * DAY, 0.5),
    ], columns=["event_id", "visitorid", "itemid", "behavior",
                "timestamp", "w_event"]).to_csv(out / "events.csv", index=False)
    pd.DataFrame([
        (1, 10, 2.7, 1, 1, 0, 95 * DAY),
        (1, 11, 0.95, 1, 0, 0, 99 * DAY),
        (2, 10, 0.5, 1, 0, 0, 50 * DAY),
    ], columns=["visitor_id", "item_id", "weight", "n_view", "n_cart",
                "n_txn", "last_ts"]).to_csv(out / "interacted_with.csv", index=False)
    pd.DataFrame([(10, 0), (11, 1), (12, 2)],
                 columns=["item_id", "idx"]).to_csv(out / "items.csv", index=False)
    pd.DataFrame([(1, 0), (2, 1)],
                 columns=["visitor_id", "idx"]).to_csv(out / "visitors.csv", index=False)
    pd.DataFrame([(10, 500, 0, 1.0), (12, 500, 0, 1.0), (11, 600, 0, 1.0)],
                 columns=["item_id", "category_id", "valid_from", "rel_weight"]
                 ).to_csv(out / "has_category.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({"cohort": "original"}), encoding="utf-8")
    return out


# ── Nap ──────────────────────────────────────────────────────────────────


def test_a_missing_export_says_which_command_makes_it(tmp_path) -> None:
    """Thong bao loi phai chi duong, khong chi bao 'khong tim thay'."""
    with pytest.raises(TraceUnavailableError, match="make neo4j"):
        load_trace(tmp_path / "chua-co")


def test_trace_loads_and_lists_its_visitors(export_dir) -> None:
    trace = load_trace(export_dir)

    assert trace.visitor_ids == [1, 2]
    assert len(trace.events) == 4


def test_a_missing_topk_points_at_drive(tmp_path) -> None:
    """topk.csv bi gitignore vi nang — loi phai noi cho lay o dau."""
    run = tmp_path / "original_bt_dkgrec_2020_20260101-000000"
    run.mkdir()

    with pytest.raises(TraceUnavailableError, match="Drive"):
        load_topk(run)


def test_a_topk_missing_a_column_is_refused(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame([(1, 1, 10)], columns=["visitor_id", "rank", "item_id"]).to_csv(
        run / "topk.csv", index=False)

    with pytest.raises(TraceUnavailableError, match="thieu cot"):
        load_topk(run)


# ── Truy vet mot khach ───────────────────────────────────────────────────


def test_history_is_newest_first_and_keeps_the_event_weight(export_dir) -> None:
    history = visitor_history(load_trace(export_dir), 1)

    assert list(history["w_event"]) == [0.95, 1.8, 0.9]      # moi nhat truoc
    assert list(history["behavior"]) == ["view", "addtocart", "view"]


def test_the_aggregated_edge_equals_the_sum_of_its_events(export_dir) -> None:
    """★ Day la dieu demo dung de giai thich: W(u,i) den tu dau."""
    trace = load_trace(export_dir)
    edges = visitor_edges(trace, 1)
    history = trace.events[(trace.events["visitorid"] == 1)
                           & (trace.events["itemid"] == 10)]

    weight = float(edges.loc[edges["item_id"] == 10, "weight"].iloc[0])
    assert weight == pytest.approx(history["w_event"].sum())


def test_an_unknown_visitor_yields_empty_frames_not_an_error(export_dir) -> None:
    trace = load_trace(export_dir)

    assert visitor_history(trace, 999).empty
    assert visitor_edges(trace, 999).empty


# ── Giai thich ───────────────────────────────────────────────────────────


def test_explanation_finds_the_category_path(export_dir) -> None:
    """Item 12 chua tung duoc xem, nhung chung danh muc 500 voi item 10."""
    explanation = explain_recommendation(load_trace(export_dir), 1, 12)

    assert explanation["da_tuong_tac_truc_tiep"] is False
    assert explanation["danh_muc_cua_item"] == [500]
    assert explanation["duong_noi_qua_danh_muc"] == [
        {"item_da_xem": 10, "danh_muc_chung": [500], "trong_so": pytest.approx(2.7)}
    ]


def test_explanation_orders_paths_by_edge_weight(export_dir) -> None:
    """Duong noi qua item khach quan tam nhat phai duoc neu truoc."""
    explanation = explain_recommendation(load_trace(export_dir), 1, 10)
    weights = [path["trong_so"] for path in explanation["duong_noi_qua_danh_muc"]]

    assert weights == sorted(weights, reverse=True)


# ── Rang buoc thiet ke ───────────────────────────────────────────────────


def test_the_demo_layer_never_scores_anything() -> None:
    """★ Demo phat lai topk.csv. Neu no tu cham diem, giao dien se hien so khac
    bang ket qua trong de an va khong ai biet cai nao dung.

    Kiem o muc nhap khau: khong torch, khong model, khong evaluator.
    """
    source = Path("src/export/trace.py").read_text(encoding="utf-8")

    for forbidden in ("import torch", "from src.models", "from src.evaluation",
                      "propagate", "def score"):
        assert forbidden not in source, f"trace.py khong duoc chua {forbidden!r}"


def test_recommendations_come_straight_from_the_run_artifact(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame([
        (1, "warm", 2, 20, False),
        (1, "warm", 1, 10, True),
        (2, "warm", 1, 30, False),
    ], columns=["visitor_id", "segment", "rank", "item_id", "is_target"]).to_csv(
        run / "topk.csv", index=False)

    rows = visitor_recommendations(load_topk(run), 1)

    assert list(rows["rank"]) == [1, 2]
    assert list(rows["item_id"]) == [10, 20]
    assert list(rows["is_target"]) == [True, False]


def test_the_streamlit_app_is_only_a_view() -> None:
    """★ Cung rang buoc, ap o lop hien thi.

    Neu `app/main.py` tu nhap model hay torch, gioi han "phat lai, khong cham
    diem lai" bi pha ngay tai cho de pha nhat — va bang ket qua trong de an se
    khong con la nguon su that duy nhat.
    """
    source = Path("app/main.py").read_text(encoding="utf-8")

    for forbidden in ("import torch", "from src.models", "from src.evaluation",
                      "from src.training", "from src.graph"):
        assert forbidden not in source, f"app/main.py khong duoc chua {forbidden!r}"


def test_the_app_states_its_limit_to_the_viewer() -> None:
    """Gioi han phai hien tren man hinh, khong chi nam trong docstring."""
    source = Path("app/main.py").read_text(encoding="utf-8")

    assert "phát lại" in source and "không tự chấm điểm" in source
