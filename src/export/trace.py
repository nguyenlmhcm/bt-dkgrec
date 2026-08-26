"""Doc lop truy vet cho demo (Buoc 10).

Vi sao tach khoi `app/main.py`
------------------------------
Streamlit khong test duoc bang pytest ma khong dung len mot server. Toan bo phan
CO THE SAI — doc file, ghep bang, giai thich mot goi y — nam o day, thuan pandas.
`app/main.py` chi con la lop hien thi.

Rang buoc khong duoc pha
------------------------
Demo **phat lai** `topk.csv`, khong bao gio cham diem lai (luan van muc 3.7.2).
Module nay vi vay khong nhap model, khong nhap torch, va khong co ham nao tinh
diem. Neu mot ngay nao do can "goi y truc tiep", phai la mot quyet dinh duoc ghi
lai, khong phai mot ham lang le xuat hien o day.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

#: File toi thieu de demo chay duoc, sinh boi scripts/07_export_neo4j.py.
REQUIRED_FILES = ("events.csv", "interacted_with.csv", "items.csv", "visitors.csv")

TOPK_COLUMNS = ("visitor_id", "segment", "rank", "item_id", "is_target")


class TraceUnavailableError(RuntimeError):
    """Thieu artifact de truy vet — noi ro thieu gi thay vi hien bang rong."""


@dataclass(frozen=True)
class TraceLayer:
    """Lop truy vet cua mot cohort/model, da nap san vao bo nho."""

    events: pd.DataFrame
    edges: pd.DataFrame
    items: pd.DataFrame
    visitors: pd.DataFrame
    has_category: pd.DataFrame
    manifest: dict

    @property
    def visitor_ids(self) -> list[int]:
        return sorted(self.edges["visitor_id"].unique().tolist())


def load_trace(export_dir: Path) -> TraceLayer:
    """Nap CSV do `scripts/07_export_neo4j.py` sinh ra."""
    missing = [name for name in REQUIRED_FILES if not (export_dir / name).exists()]
    if missing:
        raise TraceUnavailableError(
            f"thieu {', '.join(missing)} trong {export_dir}. "
            f"Chay: make neo4j COHORT=<cohort>"
        )
    optional = export_dir / "has_category.csv"
    manifest_path = export_dir / "manifest.json"
    import json

    return TraceLayer(
        events=pd.read_csv(export_dir / "events.csv"),
        edges=pd.read_csv(export_dir / "interacted_with.csv"),
        items=pd.read_csv(export_dir / "items.csv"),
        visitors=pd.read_csv(export_dir / "visitors.csv"),
        has_category=(pd.read_csv(optional) if optional.exists()
                      else pd.DataFrame(columns=["item_id", "category_id"])),
        manifest=(json.loads(manifest_path.read_text(encoding="utf-8"))
                  if manifest_path.exists() else {}),
    )


def load_topk(run_dir: Path) -> pd.DataFrame:
    """Doc `topk.csv` cua mot run artifact.

    File nay bi gitignore (nang) va chi nam tren Drive, nen thong bao loi phai
    noi ro cach lay ve — nguoi dung se gap no truoc khi gap giao dien.
    """
    path = run_dir / "topk.csv"
    if not path.exists():
        raise TraceUnavailableError(
            f"khong thay {path}. File nay bi gitignore vi nang; chep tu Drive: "
            f"MyDrive/bt-dkgrec/runs/{run_dir.name}/topk.csv"
        )
    frame = pd.read_csv(path)
    missing = [c for c in TOPK_COLUMNS if c not in frame.columns]
    if missing:
        raise TraceUnavailableError(f"{path} thieu cot: {missing}")
    assert_topk_belongs_to_run(frame, run_dir)
    return frame


def assert_topk_belongs_to_run(topk: pd.DataFrame, run_dir: Path) -> None:
    """`topk.csv` phai la cua CHINH run nay.

    File nay bi gitignore va duoc chep tay tu Drive, nen viec chep nham run —
    hoac nham ca cohort — la chuyen se xay ra, khong phai co the xay ra. Neu
    khong chan, demo hien goi y cua mot cohort trong khi thanh ben ghi ten cohort
    khac, va khong co dau hieu nao tren man hinh de ai do nhan ra.

    Doi chieu voi `metrics.json` cua chinh run do: so nguoi dung duoc xep hang
    khong duoc vuot qua so nguoi dung run nay danh gia.
    """
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return                                  # khong co gi de doi chieu

    import json

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    counts = (payload.get("test") or {}).get("n_users") or {}
    expected = counts.get("all")
    if expected is None:
        return

    actual = int(topk["visitor_id"].nunique())
    if actual > int(expected):
        raise TraceUnavailableError(
            f"topk.csv khong phai cua run nay: file co {actual:,} nguoi dung nhung "
            f"{run_dir.name} chi danh gia {int(expected):,} "
            f"(cohort {payload.get('cohort')}). Chep dung file tu Drive: "
            f"MyDrive/bt-dkgrec/runs/{run_dir.name}/topk.csv"
        )


def visitor_history(trace: TraceLayer, visitor_id: int) -> pd.DataFrame:
    """Tung su kien cua mot khach, moi nhat truoc, kem trong so w_event."""
    history = trace.events[trace.events["visitorid"] == visitor_id].copy()
    if history.empty:
        return history
    history["thoi_diem"] = pd.to_datetime(history["timestamp"], unit="ms")
    return history.sort_values("timestamp", ascending=False)[
        ["thoi_diem", "itemid", "behavior", "w_event"]
    ]


def visitor_edges(trace: TraceLayer, visitor_id: int) -> pd.DataFrame:
    """Canh da gop `W(u,i)` — chinh la thu mo hinh hoc tren do."""
    edges = trace.edges[trace.edges["visitor_id"] == visitor_id].copy()
    if edges.empty:
        return edges
    edges["lan_cuoi"] = pd.to_datetime(edges["last_ts"], unit="ms")
    return edges.sort_values("weight", ascending=False)[
        ["item_id", "weight", "n_view", "n_cart", "n_txn", "lan_cuoi"]
    ]


def visitor_recommendations(topk: pd.DataFrame, visitor_id: int) -> pd.DataFrame:
    """Top-K da phat lai tu run artifact. KHONG cham diem lai."""
    rows = topk[topk["visitor_id"] == visitor_id]
    return rows.sort_values("rank")[["rank", "item_id", "is_target"]]


def explain_recommendation(
    trace: TraceLayer, visitor_id: int, item_id: int
) -> dict[str, object]:
    """Vi sao item nay xuat hien — bang bang chung tu do thi, khong phai bang diem.

    Ba duong noi duoc kiem tra, theo dung ba loai canh cua do thi:

    * khach da tuong tac truc tiep voi item (hiem, vi item da xem bi loc);
    * item chung danh muc voi thu khach tung xem;
    * khach tung xem item nao trong cung danh muc, va nang bao nhieu.

    Tra ve bang chung tho. Dien giai thanh cau chu la viec cua lop hien thi.
    """
    history_items = set(trace.edges.loc[
        trace.edges["visitor_id"] == visitor_id, "item_id"
    ])
    direct = item_id in history_items

    categories_of = trace.has_category.groupby("item_id")["category_id"].apply(set)
    item_categories = categories_of.get(item_id, set())
    shared: list[dict[str, object]] = []
    if item_categories:
        weights = trace.edges[trace.edges["visitor_id"] == visitor_id]
        weights = weights.set_index("item_id")["weight"]
        for other in history_items:
            common = item_categories & categories_of.get(other, set())
            if common:
                shared.append({
                    "item_da_xem": int(other),
                    "danh_muc_chung": sorted(int(c) for c in common),
                    "trong_so": float(weights.get(other, 0.0)),
                })
    shared.sort(key=lambda r: r["trong_so"], reverse=True)

    return {
        "visitor_id": int(visitor_id),
        "item_id": int(item_id),
        "da_tuong_tac_truc_tiep": bool(direct),
        "danh_muc_cua_item": sorted(int(c) for c in item_categories),
        "duong_noi_qua_danh_muc": shared,
    }
