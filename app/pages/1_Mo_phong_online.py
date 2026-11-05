"""Mo phong online theo thiet ke muc 3.7.1: bam hanh vi, Top-K tinh lai ngay (D38)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.serving.service import DemoService, event_weight, rank_change  # noqa: E402
from src.serving.store import GraphStore  # noqa: E402

COHORT, MODEL = "active", "bt_dkgrec_l05"
SERVING = ROOT / "experiments" / "serving"
BEHAVIOR_LABEL = {"view": "Xem", "addtocart": "Thêm giỏ", "transaction": "Mua"}
DEFAULT_VISITOR = 962057

st.set_page_config(page_title="Mô phỏng online", layout="wide")
st.markdown('<style>[data-testid="stButton"] button p {white-space: nowrap;}</style>', unsafe_allow_html=True)


def _latest_e0() -> Path | None:
    files = sorted(SERVING.glob(f"{COHORT}_{MODEL}_*/e0_best.npy"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


@st.cache_resource(show_spinner="Đang dựng dịch vụ dự báo…")
def _service(path: str, _mtime: float) -> DemoService:
    return DemoService.from_artifacts(COHORT, MODEL, Path(path))


@st.cache_resource
def _store() -> GraphStore:
    return GraphStore()


st.title("Mô phỏng online")
e0_path = _latest_e0()
if e0_path is None:
    st.error("Chưa có model trong `experiments/serving/`.")
    st.stop()
try:
    store = _store()
    n_events = store.ping()
except Exception as error:  # noqa: BLE001 - shown to the operator as-is
    st.error(f"Không kết nối được Neo4j. Chạy `bash scripts/23_neo4j_demo.sh`.\n\n{error}")
    st.stop()

service = _service(str(e0_path), e0_path.stat().st_mtime)
meta = json.loads((e0_path.parent / "e0_best.json").read_text(encoding="utf-8"))
st.caption(
    "Bấm hành vi → ghi vào Neo4j → đọc lại lịch sử từ Neo4j → dịch vụ dự báo tính lại Top-K.  \n"
    f"Nhóm Active, BT-DKGRec-GCN (λ=0.05), bản huấn luyện riêng cho demo: NDCG@20 tập xác thực = {meta['value']:.4f}. "
    "Đây là minh họa luồng hệ thống; số chính thức của đề án nằm ở Chương 4."
)


@st.cache_data(show_spinner=False)
def _visitors() -> list[int]:
    with (ROOT / "data" / "neo4j" / f"{COHORT}_{MODEL}_evaluated" / "visitors.csv").open(encoding="utf-8") as f:
        return pd.read_csv(f)["visitor_id"].astype(int).tolist()


visitors = _visitors()
visitor = st.sidebar.selectbox(
    "Khách hàng", visitors, index=visitors.index(DEFAULT_VISITOR) if DEFAULT_VISITOR in visitors else 0
)
if st.sidebar.button("Hoàn tác hành vi đã bấm"):
    st.sidebar.success(f"Đã xóa {store.reset(visitor)} sự kiện mô phỏng")
    st.session_state.pop(f"truoc_{visitor}", None)
    st.session_state.pop(f"vua_bam_{visitor}", None)
st.sidebar.caption(f"Kho đồ thị Neo4j: {n_events:,} sự kiện".replace(",", "."))

history = store.history(visitor)
started = time.perf_counter()
top = service.recommend(visitor, history, k=10)
ms = (time.perf_counter() - started) * 1000

left, right = st.columns([3, 2])
truoc = st.session_state.get(f"truoc_{visitor}")
with left:
    st.subheader("Gợi ý hiện tại")
    vua_bam = st.session_state.get(f"vua_bam_{visitor}")
    if vua_bam:
        st.success(vua_bam)
    st.caption(f"Tính lại từ lịch sử trong Neo4j mất {ms:.0f} ms")
    for row in top.itertuples():
        a, b, c, d, e = st.columns([0.5, 2.4, 1, 1.5, 1], vertical_alignment="center")
        a.write(f"**{row.rank}**")
        doi = rank_change(truoc, int(row.item_id), int(row.rank)) if truoc is not None else ""
        mau = "gray" if doi.startswith("▼") else "red"
        b.write(f"Sản phẩm {row.item_id}" + (f" &nbsp; :{mau}[**{doi}**]" if doi else ""))
        for col, behavior in zip((c, d, e), BEHAVIOR_LABEL):
            if col.button(BEHAVIOR_LABEL[behavior], key=f"{behavior}_{row.item_id}"):
                st.session_state[f"truoc_{visitor}"] = top["item_id"].astype(int).tolist()
                w = event_weight(service.bundle, behavior)
                store.ingest(visitor, int(row.item_id), behavior, service.next_timestamp(history), w)
                st.session_state[f"vua_bam_{visitor}"] = (
                    f"Đã ghi vào Neo4j: {BEHAVIOR_LABEL[behavior]} sản phẩm {row.item_id}, w = {w:g}"
                )
                st.rerun()
with right:
    st.subheader("Lịch sử trong Neo4j")
    st.metric("Sự kiện", f"{len(history):,}", delta=f"{int(history['demo'].sum())} vừa bấm" if history["demo"].any() else None)
    shown = history.sort_values("timestamp", ascending=False).head(15).copy()
    shown["behavior"] = shown["behavior"].map(BEHAVIOR_LABEL)
    shown["demo"] = shown["demo"].map({True: "vừa bấm", False: ""})
    st.dataframe(shown[["item_id", "behavior", "w", "demo"]].rename(
        columns={"item_id": "Sản phẩm", "behavior": "Hành vi", "demo": ""}), hide_index=True, width="stretch")
