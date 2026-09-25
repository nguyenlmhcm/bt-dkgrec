"""Va v16 -> v17: bo dau +/-, bo bang gia tri p, thong nhat mo hinh de xuat la lambda=0.05.

Ba khiem khuyet duoc sua o day deu cung mot goc: tai lieu dang viet o muc paper
hoi nghi chu khong phai muc de an ung dung ma truong da duyet.

1. Dau +/- xuat hien 75 lan. Do tren hai bai mau HUIT ("de an Khang - form
   chuan.docx", "NGUYENTHIBICHTUYEN.pdf"): CA HAI deu 0 lan. Bang 4.7 them o
   buoc 14 da liet ke tung seed, nen +/- thanh thua -- bo di khong mat thong tin.

2. Bang 4.9 la bang gia tri p (Welch / ghep cap theo seed); Bang 4.10 va 4.11
   con hai cot Welch va Ghep cap. Nguoi dung da bac bo cach viet nay ("2 bai toi
   gui ban tham khao co ai viet p gi dau"), buoc truoc chi sua loi dan ma bo sot
   bang. Do lai: p-value = 0 lan o ca hai bai mau.

3. Nghiem trong nhat: Bang 4.8 so LightGCN voi lambda=0.01 (+21,24% / +13,44%)
   trong khi Tom tat khang dinh 28% / 33% -- do la so cua lambda=0.05. Mau thuan
   trong cung mot quyen. Va cau "lan chay thap nhat van cao hon lan chay tot nhat"
   CHI dung voi lambda=0.05: voi lambda=0.01, 0.023102 < 0.025990 (chong lan).
   Bang 3.6 do lambda tren tap xac thuc va chon 0.05; muc 4.2.4 ghi ro 0.01 la
   "gia tri ke thua va chua qua buoc do". Nen mo hinh de xuat la lambda=0.05.

Chay:
    python scripts/15_bo_pm.py
    python scripts/15_bo_pm.py --no-highlight --out docs/De_an_thac_si_v17_ban_nop.docx
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)

SOURCE = Path("docs/De_an_thac_si_v16.docx")
RUNS = Path("experiments/runs")
OMML = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

TOUCHED: list = []
CHANGELOG: list[tuple[str, str, str]] = []

#: Sau khi xoa Bang 4.9, moi bang tu 4.10 tro di lui mot so.
LUI_SO = {"4.10": "4.9", "4.11": "4.10", "4.12": "4.11", "4.13": "4.12"}

CAPTION_XOA = "Bảng 4.9. Kiểm định ý nghĩa thống kê, BT-DKGRec-GCN so với LightGCN"

#: Cot bi go khoi Bang 4.10 va 4.11 -- khop theo tieu de cot, khong theo chi so.
COT_XOA = {"Welch", "Ghép cặp", "Ghép cặp theo seed"}

CHI_SO = [("recall@20", "Recall@20"),
          ("ndcg@20", "NDCG@20"),
          ("hit_rate@20", "HitRate@20")]

SPLITS = [("original", "original"), ("active", "active")]


def doc_so(runs: Path) -> dict:
    out: dict = {}
    for f in glob.glob(str(runs / "*" / "metrics.json")):
        m = json.loads(Path(f).read_text())
        w = m["test"]["warm"]
        for khoa, _ in CHI_SO:
            out.setdefault((m["cohort"], m["model"], khoa), {})[m["seed"]] = w[khoa]
    if not out:
        raise SystemExit(f"LOI: khong doc duoc metrics.json trong {runs}")
    return out


# ── Tien ich ────────────────────────────────────────────────────────────


def para_khop(doc: Document, text: str):
    hits = [p for p in doc.paragraphs
            if p.text.strip() == text and p.style.name != "List Bullet"]
    if len(hits) != 1:
        raise SystemExit(f"LOI: doan {text[:60]!r} khop {len(hits)} lan, can 1.")
    return hits[0]


def dat_text(para, text: str) -> None:
    if not para.runs:
        para.add_run(text)
    else:
        para.runs[0].text = text
        for run in para.runs[1:]:
            run._r.getparent().remove(run._r)
    TOUCHED.append(para)


def bang_theo_caption(doc: Document) -> dict[str, Table]:
    """Gan moi bang voi caption dung truoc no trong than tai lieu."""
    out: dict[str, Table] = {}
    last = None
    for el in doc.element.body:
        if el.tag == qn("w:p"):
            t = Paragraph(el, doc).text.strip()
            if re.match(r"^Bảng \d+\.\d+\.", t):
                last = t
        elif el.tag == qn("w:tbl") and last is not None:
            out.setdefault(last, Table(el, doc))
            last = None
    return out


def xoa_cot(table: Table, ten_cot: set[str]) -> int:
    """Go cac cot co tieu de nam trong `ten_cot`, ke ca o w:tblGrid."""
    idx = [i for i, c in enumerate(table.rows[0].cells) if c.text.strip() in ten_cot]
    if not idx:
        return 0
    for row in table.rows:
        tcs = row._tr.findall(qn("w:tc"))
        for i in sorted(idx, reverse=True):
            if i < len(tcs):
                row._tr.remove(tcs[i])
    grid = table._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        cols = grid.findall(qn("w:gridCol"))
        for i in sorted(idx, reverse=True):
            if i < len(cols):
                grid.remove(cols[i])
    return len(idx)


# ── Cac buoc va ─────────────────────────────────────────────────────────


def bo_dau_pm(doc: Document) -> int:
    """Bo '+/- sd' khoi moi o bang. Bang 4.7 da giu phan bien thien."""
    n = 0
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        moi = re.sub(r"\s*±\s*[\d.]+", "", run.text)
                        if moi != run.text:
                            run.text = moi
                            n += 1
                    TOUCHED.append(para)
    for para in doc.paragraphs:
        if "±" in para.text:
            dat_text(para, para.text.replace(", trung bình ± độ lệch chuẩn trên ba seed",
                                             ", trung bình trên ba seed"))
            n += 1
    if n == 0:
        raise SystemExit("LOI: khong tim thay dau ± nao; nguon co dung v16 khong?")
    CHANGELOG.append((
        "Chương 4, các bảng kết quả",
        "Bỏ ký hiệu ± độ lệch chuẩn, caption đổi thành “trung bình trên ba seed”",
        "Hai bài mẫu của trường dùng ± 0 lần; Bảng 4.7 đã liệt kê từng seed nên "
        "phần biến thiên không mất"))
    return n


def doi_bang_48(doc: Document, so: dict) -> None:
    """Bang 4.8 dang so lambda=0.01; Tom tat noi ve lambda=0.05."""
    caption = [p for p in doc.paragraphs
               if p.text.strip().startswith("Bảng 4.8.")
               and p.style.name != "List Bullet"]
    if len(caption) != 1:
        raise SystemExit(f"LOI: caption Bang 4.8 khop {len(caption)} lan.")
    table = bang_theo_caption(doc)[caption[0].text.strip()]

    dat_text(table.rows[0].cells[3].paragraphs[0], "BT-DKGRec-GCN (λ=0,05)")
    dong = 1
    for cohort, nhan in SPLITS:
        for khoa, nhan_cs in CHI_SO:
            lgn = so[(cohort, "lightgcn", khoa)]
            bt = so[(cohort, "bt_dkgrec_l05", khoa)]
            lm, bm = st.mean(lgn.values()), st.mean(bt.values())
            thang = sum(1 for s in lgn if bt[s] > lgn[s])
            cells = table.rows[dong].cells
            for cell, text in zip(cells, [nhan, nhan_cs, f"{lm:.6f}", f"{bm:.6f}",
                                          f"{(bm / lm - 1) * 100:+.2f}%",
                                          f"{thang}/3"]):
                cell.text = str(text)
                TOUCHED.append(cell.paragraphs[0])
            dong += 1
    CHANGELOG.append((
        "Chương 4, Bảng 4.8",
        "Đổi cột mô hình đề xuất từ λ=0,01 sang λ=0,05",
        "Bảng cũ ghi +21,24% và +13,44% (λ=0,01) trong khi Tóm tắt nêu 28% và 33% "
        "(λ=0,05); λ=0,05 mới là giá trị đã dò trên tập xác thực theo Bảng 3.6"))


def xoa_bang_49(doc: Document) -> None:
    cap = para_khop(doc, CAPTION_XOA)
    table = bang_theo_caption(doc).get(CAPTION_XOA)
    if table is None:
        raise SystemExit("LOI: khong thay bang di kem caption 4.9.")
    el = cap._p.getnext()
    table._tbl.getparent().remove(table._tbl)
    cap._p.getparent().remove(cap._p)
    if el is not None and el.tag == qn("w:p") and not Paragraph(el, doc).text.strip():
        el.getparent().remove(el)
    CHANGELOG.append((
        "Chương 4, mục 4.3.2",
        "Xoá Bảng 4.9 (giá trị p của kiểm định Welch và ghép cặp theo seed)",
        "Hai bài mẫu của trường không có bảng giá trị p nào"))


def go_cot_thong_ke(doc: Document) -> None:
    tong = 0
    for cap, table in bang_theo_caption(doc).items():
        if cap.startswith(("Bảng 4.10.", "Bảng 4.11.")):
            n = xoa_cot(table, COT_XOA)
            tong += n
            if n:
                TOUCHED.append(table)
    if tong == 0:
        raise SystemExit("LOI: khong go duoc cot Welch/Ghep cap nao.")
    CHANGELOG.append((
        "Chương 4, Bảng 4.10 và 4.11",
        f"Gỡ {tong} cột giá trị p (Welch, Ghép cặp), giữ cột “Seed thắng”",
        "Cột “Seed thắng” diễn đạt cùng ý bằng lời thường và đọc được ngay"))


def lui_so_bang(doc: Document) -> None:
    """Sau khi xoa Bang 4.9, cac bang sau lui mot so. Mot luot duy nhat."""
    mau = re.compile(r"Bảng (4\.1[0-3])\b")

    def thay(m):
        return "Bảng " + LUI_SO[m.group(1)]

    n = 0
    for para in doc.paragraphs:
        if mau.search(para.text):
            dat_text(para, mau.sub(thay, para.text))
            n += 1
    if n == 0:
        raise SystemExit("LOI: khong thay so hieu 4.10-4.13 nao de lui.")
    CHANGELOG.append((
        "Chương 4, đánh số bảng",
        "Bảng 4.10→4.9, 4.11→4.10, 4.12→4.11, 4.13→4.12",
        "Đánh số lại cho liên tục sau khi xoá Bảng 4.9"))


def viet_lai_loi_dan(doc: Document, so: dict) -> None:
    """Ba doan van dang dua ket luan vao gia tri p hoac noi sai so seed thang."""
    r_o = so[("original", "bt_dkgrec_l05", "recall@20")]
    l_o = so[("original", "lightgcn", "recall@20")]

    dat_text(para_khop(
        doc,
        "BT-DKGRec-GCN cao hơn LightGCN ở cả ba chỉ số xếp hạng trên cả hai split, "
        "và cao hơn ở cả ba seed trong từng phép so sánh."),
        "BT-DKGRec-GCN cao hơn LightGCN ở cả ba chỉ số xếp hạng trên cả hai nhóm "
        "người dùng. Ở Recall@20 và HitRate@20, mô hình đề xuất cao hơn trong cả ba "
        "lần chạy; ở NDCG@20, mô hình đề xuất cao hơn trong hai trong ba lần chạy.")

    dat_text(para_khop(
        doc,
        "Mức cải thiện được ghi nhận ở cả ba lần chạy trên cả hai nhóm người dùng, "
        "không có lần chạy nào cho kết quả ngược lại. Bảng 4.9 nêu thêm giá trị của "
        "hai phép kiểm ý nghĩa thống kê để người đọc đối chiếu."),
        f"Khoảng cách giữa hai mô hình lớn hơn mức dao động giữa các lần chạy. Theo "
        f"Bảng 4.7, lần chạy cho kết quả cao nhất của LightGCN ở Recall@20 đạt "
        f"{max(l_o.values()):.6f}, vẫn thấp hơn lần chạy cho kết quả thấp nhất của mô "
        f"hình đề xuất là {min(r_o.values()):.6f}. Điều tương tự đúng trên nhóm người "
        f"dùng tích cực.")

    cu = para_khop(
        doc,
        "Hai phép kiểm được nêu song song và không phép kiểm nào được chọn sau khi "
        "quan sát kết quả. Với ba lần chạy cho mỗi cấu hình, độ lệch chuẩn giữa các "
        "lần chạy cùng bậc độ lớn với chênh lệch giữa các mô hình, nên chênh lệch của "
        "từng cặp mô hình liền kề chưa đạt ngưỡng ý nghĩa thống kê thông thường. Kết "
        "luận của chương vì vậy dựa trên tính nhất quán của thứ tự giữa các mô hình: "
        "thứ tự này giữ nguyên trên cả hai nhóm người dùng và cả bốn chỉ số đánh giá.")
    dat_text(cu,
             "Với ba lần chạy cho mỗi cấu hình, kết luận của chương dựa trên tính "
             "nhất quán của thứ tự giữa các mô hình: thứ tự này giữ nguyên trên cả "
             "hai nhóm người dùng và cả bốn chỉ số đánh giá.")

    dat_text(para_khop(
        doc,
        "Không giá trị p nào trong nhóm so sánh này đạt mức 0.05. Trên bộ dữ liệu "
        "này, đề án chưa chứng minh được behavior-time weighting cải thiện độ chính "
        "xác xếp hạng so với knowledge graph tĩnh."),
        "Khoảng cách giữa hai mô hình trong nhóm so sánh này nhỏ hơn mức dao động "
        "giữa các lần chạy. Trên bộ dữ liệu này, đề án chưa chứng minh được rằng "
        "riêng trọng số hành vi-thời gian cải thiện độ chính xác xếp hạng so với đồ "
        "thị tri thức tĩnh.")

    CHANGELOG.append((
        "Chương 4, mục 4.3.2 và 4.3.3",
        "Viết lại bốn đoạn dẫn: bỏ dẫn chiếu giá trị p, nêu đúng số lần chạy thắng",
        "Đoạn cũ nói “cao hơn ở cả ba seed trong từng phép so sánh” trong khi NDCG@20 "
        "chỉ thắng 2/3; và ba đoạn còn lại dựa kết luận vào giá trị p"))


def cap_nhat_danh_muc(doc: Document) -> None:
    cu = [p for p in doc.paragraphs
          if p.style.name == "List Bullet" and p.text.strip() == CAPTION_XOA]
    for p in cu:
        p._p.getparent().remove(p._p)
    log.info("Da bo %d muc Bang 4.9 khoi danh muc bang", len(cu))


def xoa_muc_changelog(doc: Document) -> int:
    """Go trang "DANH MUC SUA DOI SO VOI BAN v11" khoi ban nop.

    Do la cong cu ra soat noi bo: no doi chieu voi mot ban nhap chua tung nop.
    No cung la cho duy nhat con trich lai ky hieu ± va cum "gia tri p" sau khi
    than bai da sach.

    CHi go tieu de, doan dan, va DUNG bang nhat ky. Khong duoc di tiep toi tieu
    de ke sau: giua chung la HAI BANG TRANG BIA ("BO CONG THUONG ...", "CONG HOA
    XA HOI CHU NGHIA VIET NAM"), mot trong do chua logo truong.
    """
    head = [p for p in doc.paragraphs
            if p.text.strip() == "DANH MỤC SỬA ĐỔI SO VỚI BẢN v11"]
    if not head:
        return 0
    n, el, thay_bang = 0, head[0]._p, False
    while el is not None and not thay_bang:
        ke = el.getnext()
        if el.tag == qn("w:tbl"):
            dau = Table(el, doc).rows[0].cells[0].text
            if "Mục" not in dau:
                break          # da di qua bang nhat ky -- dung ngay, giu trang bia
            thay_bang = True
        el.getparent().remove(el)
        n += 1
        el = ke
    if not thay_bang:
        raise SystemExit("LOI: khong thay bang nhat ky sua doi de go.")
    return n


def append_changelog(doc: Document) -> None:
    tables = [t for t in doc.tables if t.rows and "Mục" in t.rows[0].cells[0].text]
    if len(tables) != 1:
        log.warning("Khong thay bang danh muc sua doi (%d khop); bo qua.", len(tables))
        return
    for muc, sua, ly_do in CHANGELOG:
        cells = tables[0].add_row().cells
        for cell, text in zip(cells, (f"[v17] {muc}", sua, ly_do)):
            cell.text = text
        TOUCHED.append(tables[0])


def highlight_touched() -> int:
    n = 0
    for item in TOUCHED:
        is_table = hasattr(item, "rows")
        paras = ([p for r in item.rows for c in r.cells for p in c.paragraphs]
                 if is_table else [item])
        for para in paras:
            for run in para.runs:
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
                n += 1
    return n


def xoa_to_nen(doc: Document) -> int:
    n = 0

    def quet(paras):
        nonlocal n
        for para in paras:
            for run in para.runs:
                if run.font.highlight_color is not None:
                    run.font.highlight_color = None
                    n += 1

    quet(doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                quet(cell.paragraphs)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--out", type=Path, default=Path("docs/De_an_thac_si_v17.docx"))
    ap.add_argument("--no-highlight", action="store_true")
    ap.add_argument("--bo-changelog", action="store_true",
                    help="go trang danh muc sua doi -- dung cho ban nop")
    args = ap.parse_args()

    if not args.source.exists():
        raise SystemExit(f"LOI: khong thay {args.source}")

    so = doc_so(args.runs)
    doc = Document(str(args.source))
    ct_truoc = len(doc.element.body.findall(f".//{OMML}oMath"))
    bang_truoc, hinh_truoc = len(doc.tables), len(doc.inline_shapes)

    # Thu tu bat buoc: doi so lieu va go cot TRUOC khi lui so hieu, vi cac ham
    # tren tim bang theo caption cu.
    doi_bang_48(doc, so)
    viet_lai_loi_dan(doc, so)
    go_cot_thong_ke(doc)
    xoa_bang_49(doc)
    cap_nhat_danh_muc(doc)
    lui_so_bang(doc)
    log.info("Bo dau ± o %d cho", bo_dau_pm(doc))
    append_changelog(doc)

    # Bang danh muc sua doi TRICH LAI ky hieu vua go ("Bo ky hieu ± ...") --
    # do la viec cua no, khong phai dau con sot.
    sot = []
    for i, table in enumerate(doc.tables):
        if table.rows and "Mục" in table.rows[0].cells[0].text:
            continue
        for row in table.rows:
            for cell in row.cells:
                if "±" in cell.text:
                    sot.append(f"bang#{i}: {cell.text[:50]}")
    sot += [f"doan: {p.text[:70]}" for p in doc.paragraphs if "±" in p.text]
    if sot:
        raise SystemExit("LOI: con dau ± o:\n  " + "\n  ".join(sot))

    ct_sau = len(doc.element.body.findall(f".//{OMML}oMath"))
    if ct_sau != ct_truoc:
        raise SystemExit(f"LOI: cong thuc {ct_truoc} -> {ct_sau}.")
    if len(doc.inline_shapes) != hinh_truoc:
        raise SystemExit(f"LOI: hinh {hinh_truoc} -> {len(doc.inline_shapes)}.")
    if len(doc.tables) != bang_truoc - 1:
        raise SystemExit(f"LOI: bang {bang_truoc} -> {len(doc.tables)}, can -1.")

    log.info("cong thuc %d · bang %d (truoc %d) · hinh %d",
             ct_sau, len(doc.tables), bang_truoc, len(doc.inline_shapes))

    if args.bo_changelog:
        log.info("Go trang danh muc sua doi: %d phan tu", xoa_muc_changelog(doc))
    if args.no_highlight:
        log.info("Xoa to nen o %d run", xoa_to_nen(doc))
    else:
        log.info("To nen vang %d run", highlight_touched())

    doc.save(str(args.out))
    log.info("Da ghi %s", args.out)


if __name__ == "__main__":
    main()
