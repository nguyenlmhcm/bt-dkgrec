"""Demo BT-DKGRec-GCN — truy vet mot khach hang that (Buoc 10).

Chay:
    make app

Gioi han co chu y (luan van muc 3.7.2)
--------------------------------------
Ung dung nay **khong tu cham diem**. No phat lai `topk.csv` cua run artifact va
doc lop truy vet da xuat sang CSV. Nho vay con so tren man hinh luon trung voi
bang ket qua trong de an -- neu app tu suy luan, no se sinh ra mot bo so thu hai
va khong ai biet bo nao dung.

Toan bo logic nam o `src/export/trace.py` va duoc pytest bao ve. File nay chi
hien thi.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.export.trace import (  # noqa: E402
    TraceUnavailableError,
    explain_recommendation,
    load_topk,
    load_trace,
    visitor_edges,
    visitor_history,
    visitor_recommendations,
)

BEHAVIOR_LABEL = {"view": "xem", "addtocart": "thêm giỏ", "transaction": "mua"}
DEFAULT_EXPORT = "original_bt_dkgrec_l05_evaluated"
DEFAULT_VISITOR = 962057
NO_RUN = "— không nạp —"

st.set_page_config(page_title="BT-DKGRec-GCN — Demo truy vết", layout="wide")


@st.cache_data(show_spinner="Đang nạp đồ thị truy vết…")
def _trace(export_dir: str):
    return load_trace(Path(export_dir))


@st.cache_data(show_spinner="Đang nạp Top-K…")
def _topk(run_dir: str) -> pd.DataFrame:
    return load_topk(Path(run_dir))


def _discover(pattern: str, root: str) -> list[str]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(str(p) for p in base.glob(pattern) if p.is_dir())


def _runs_for(export_dir: str) -> list[str]:
    """Runs of the same cohort and model as the export that have a local topk.csv."""
    prefix = Path(export_dir).name.removesuffix("_evaluated")
    pattern = re.compile(rf"{re.escape(prefix)}_\d{{4}}_")
    return [r for r in _discover("*", "experiments/runs")
            if pattern.match(Path(r).name) and (Path(r) / "topk.csv").exists()]


def _name(path: str) -> str:
    return Path(path).name


# ── Thanh ben ────────────────────────────────────────────────────────────

st.sidebar.title("BT-DKGRec-GCN")
st.sidebar.caption("Dự báo hành vi khách hàng bằng đồ thị tri thức động")

exports = _discover("*", "data/neo4j")
if not exports:
    st.error("Chưa có bản xuất nào trong `data/neo4j/`. Chạy: `make neo4j COHORT=original`")
    st.stop()

names = [Path(e).name for e in exports]
export_dir = st.sidebar.selectbox(
    "Đồ thị đã xuất", exports, format_func=_name,
    index=names.index(DEFAULT_EXPORT) if DEFAULT_EXPORT in names else 0,
)
runs = _runs_for(export_dir)
run_dir = st.sidebar.selectbox(
    "Lần chạy (Top-K)", [*runs, NO_RUN], format_func=lambda r: r if r == NO_RUN else _name(r)
)

try:
    trace = _trace(export_dir)
except TraceUnavailableError as error:
    st.error(str(error))
    st.stop()

topk = None
if run_dir != NO_RUN:
    try:
        topk = _topk(run_dir)
    except TraceUnavailableError as error:
        st.sidebar.warning(str(error))

visitors = list(trace.visitor_ids)
visitor_id = st.sidebar.selectbox(
    "Khách hàng", visitors,
    index=visitors.index(DEFAULT_VISITOR) if DEFAULT_VISITOR in visitors else 0,
)

manifest = trace.manifest
if manifest:
    st.sidebar.divider()
    st.sidebar.caption(
        f"Nhóm **{manifest.get('cohort')}** · mô hình **{manifest.get('model')}**"
    )
    check = manifest.get("layer_consistency", {})
    if check:
        st.sidebar.caption(
            f"Hai đồ thị khớp: {check['n_pairs']:,}".replace(",", ".")
            + f" cặp, lệch tối đa {check['worst_relative_difference']:g}"
        )

# ── Nội dung ─────────────────────────────────────────────────────────────

st.title(f"Khách hàng {visitor_id}")
st.caption(
    "Trang này **phát lại** kết quả đã lưu của lần chạy, không tự chấm điểm — "
    "nên mọi con số ở đây trùng với bảng kết quả trong đề án (mục 3.7.2)."
)

history = visitor_history(trace, visitor_id)
edges = visitor_edges(trace, visitor_id)

left, middle, right = st.columns(3)
left.metric("Sự kiện", f"{len(history):,}")
middle.metric("Sản phẩm đã chạm", f"{len(edges):,}")
right.metric("Tổng trọng số W(u,·)", f"{edges['weight'].sum():.3f}" if len(edges) else "0")

tab_trace, tab_projected, tab_topk = st.tabs(
    ["Đồ thị truy vết — từng sự kiện", "Đồ thị huấn luyện — W(u,i)", "Gợi ý Top-K"]
)

with tab_trace:
    st.markdown(
        "Mỗi dòng là **một sự kiện**, mang trọng số riêng "
        r"$w = \alpha_b \cdot e^{-\lambda \Delta t}$. "
        "Hành vi mạnh hơn và gần hiện tại hơn thì nặng hơn."
    )
    if history.empty:
        st.info("Khách này không có lịch sử trong tập huấn luyện.")
    else:
        shown = history.copy()
        shown["behavior"] = shown["behavior"].map(BEHAVIOR_LABEL).fillna(shown["behavior"])
        shown.columns = ["Thời điểm", "Sản phẩm", "Hành vi", "w"]
        st.dataframe(shown, width="stretch", hide_index=True)
        st.caption(
            "Tỷ lệ α = 1 : 2 : 3 cho xem / thêm giỏ / mua hiện ra ngay trong cột w "
            "khi hai sự kiện xảy ra cùng ngày."
        )

with tab_projected:
    st.markdown(
        "Đây là thứ **mô hình thực sự học trên đó**: mỗi cặp (khách, sản phẩm) "
        "gộp thành **một** cạnh, trọng số là tổng các sự kiện ở tab bên cạnh."
    )
    if edges.empty:
        st.info("Không có cạnh nào.")
    else:
        shown = edges.copy()
        shown.columns = ["Sản phẩm", "W(u,i)", "Xem", "Thêm giỏ", "Mua", "Lần cuối"]
        st.dataframe(shown, width="stretch", hide_index=True)
        st.bar_chart(edges.set_index("item_id")["weight"].head(15))

with tab_topk:
    if topk is None:
        st.info(
            "Chưa nạp `topk.csv`. Chọn một lần chạy ở thanh bên. "
            "File này bị gitignore vì nặng — chép từ Drive về thư mục run."
        )
    else:
        rows = visitor_recommendations(topk, visitor_id)
        if rows.empty:
            st.warning("Khách này không có trong run đã chọn (khác cohort?).")
        else:
            hits = int(rows["is_target"].sum())
            st.metric("Trúng trong Top-K", f"{hits}/{len(rows)}")
            shown = rows.copy()
            shown["is_target"] = shown["is_target"].map({True: "✅ trúng", False: ""})
            shown.columns = ["Hạng", "Sản phẩm", "Kết quả"]
            st.dataframe(shown, width="stretch", hide_index=True)

            chosen = st.selectbox("Giải thích gợi ý nào?", rows["item_id"].tolist())
            explanation = explain_recommendation(trace, visitor_id, int(chosen))
            paths = explanation["duong_noi_qua_danh_muc"]
            if explanation["da_tuong_tac_truc_tiep"]:
                st.write("Khách **đã từng tương tác trực tiếp** với sản phẩm này.")
            if paths:
                st.write(
                    f"Nối với lịch sử qua **{len(paths)}** sản phẩm chung danh mục:"
                )
                st.dataframe(
                    pd.DataFrame(paths).rename(columns={
                        "item_da_xem": "Sản phẩm đã xem",
                        "danh_muc_chung": "Danh mục chung",
                        "trong_so": "W(u,i) của sản phẩm đó",
                    }),
                    width="stretch", hide_index=True,
                )
            elif not explanation["da_tuong_tac_truc_tiep"]:
                st.write(
                    "Không có đường nối trực tiếp qua danh mục — gợi ý đến từ lan "
                    "truyền nhiều bước trên đồ thị, không quy về một cạnh đơn lẻ."
                )
