"""Va v15 -> v16: them bang ket qua tung seed, danh so lai bang Chuong 4, viet lai Tom tat.

Ly do ton tai:

1. Tom tat cu dac con so ky thuat (25 con so, 8 gia tri thap phan bon chu so)
   trong khi hai bai mau cua truong dung 3 va 0. Xem docs/VIEC_DANG_CHO.md muc
   "Giong van".

2. Cau "muc cai thien dat gan 30%" lay dau tren cua dai. Do lai tu 36 file
   metrics.json thi khoang cai thien la +7,5% (Coverage@20, active) den +32,9%
   (Recall@20, active). Ban moi neu ro chi so nao dat 28/33%.

3. Ban moi khang dinh "lan chay thap nhat cua mo hinh de xuat van cao hon lan
   chay tot nhat cua LightGCN". Dieu do DUNG nhung than bai v15 khong co so lieu
   tung seed de chung minh -- 0.029146, 0.025990, 0.019675 deu xuat hien 0 lan.
   Bang 4.7 sinh boi script nay chinh la bang chung do.

4. Chuong 4 v15 danh so bang sai san: khong co Bang 4.3, co "Bang 4.4b", khong
   co Bang 4.7, va Bang 4.8 dung hai lan. Danh muc bang dung o 4.8 trong khi
   than bai co toi 4.12.

Chay:
    python scripts/14_bang_seed.py                    # ban ra soat, to nen vang
    python scripts/14_bang_seed.py --no-highlight \
        --out docs/De_an_thac_si_v16_ban_nop.docx     # ban nop
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml.ns import qn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)

SOURCE = Path("docs/De_an_thac_si_v15.docx")
RUNS = Path("experiments/runs")
OMML = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

TOUCHED: list = []
CHANGELOG: list[tuple[str, str, str]] = []

SEEDS = [2020, 2021, 2022]

#: Thu tu trinh bay, tu yeu den manh -- giong Bang 4.5 va 4.6.
MODEL_ORDER = [
    ("popularity", "Popularity"),
    ("recent_popularity", "Recent Popularity"),
    ("lightgcn", "LightGCN"),
    ("static_kg_gcn", "Static KG-GCN"),
    ("bt_dkgrec", "BT-DKGRec-GCN (λ=0.01)"),
    ("bt_dkgrec_l05", "BT-DKGRec-GCN (λ=0.05)"),
]

COHORTS = [
    ("original", "Có lịch sử ban đầu"),
    ("active", "Tích cực"),
]

#: Danh so lai. Khoa la caption hien co, gia tri la caption dung.
#: Chen Bang 4.7 vao sau Bang 4.6 khien moi bang tu 4.8 tro di da dung san,
#: nen chi con ba dong nay phai sua.
DOI_SO = {
    "Bảng 4.4. Cấu hình huấn luyện dùng chung cho toàn bộ ma trận thực nghiệm":
        "Bảng 4.3. Cấu hình huấn luyện dùng chung cho toàn bộ ma trận thực nghiệm",
    "Bảng 4.4b. Tham số trọng số behavior-time":
        "Bảng 4.4. Tham số trọng số behavior-time",
    "Bảng 4.8. Phạm vi kiểm chứng của ứng dụng nguyên mẫu":
        "Bảng 4.13. Phạm vi kiểm chứng của ứng dụng nguyên mẫu",
}

#: Danh muc bang Chuong 4 dung theo than bai sau khi da danh so lai.
DANH_MUC_C4 = [
    "Bảng 4.1. Môi trường và cấu hình thực nghiệm chính",
    "Bảng 4.2. Thống kê phân chia dữ liệu và đồ thị trong hai giai đoạn thực nghiệm",
    "Bảng 4.3. Cấu hình huấn luyện dùng chung cho toàn bộ ma trận thực nghiệm",
    "Bảng 4.4. Tham số trọng số behavior-time",
    "Bảng 4.5. Kết quả test trên nhóm người dùng có lịch sử ban đầu tại K = 20",
    "Bảng 4.6. Kết quả test trên nhóm người dùng tích cực tại K = 20",
    "Bảng 4.7. Recall@20 của từng lần chạy trên ba seed",
    "Bảng 4.8. BT-DKGRec-GCN so với LightGCN, test tại K = 20",
    "Bảng 4.9. Kiểm định ý nghĩa thống kê, BT-DKGRec-GCN so với LightGCN",
    "Bảng 4.10. BT-DKGRec-GCN so với Static KG-GCN, test tại K = 20",
    "Bảng 4.11. Static KG-GCN so với LightGCN, test Recall@20 tại K = 20",
    "Bảng 4.12. Phân bố bậc người dùng trong tập đánh giá cohort Original",
    "Bảng 4.13. Phạm vi kiểm chứng của ứng dụng nguyên mẫu",
]

CAPTION_47 = (
    "Bảng 4.7. Recall@20 của từng lần chạy trên ba seed, hai nhóm người dùng"
)

TOM_TAT = [
    "Trên các sàn thương mại điện tử, mỗi khách hàng chỉ tiếp cận được một phần rất "
    "nhỏ trong hàng chục nghìn sản phẩm đang được bày bán, và phần lớn lượt xem không "
    "dẫn tới hành vi mua. Vấn đề đặt ra cho doanh nghiệp là dự báo được sản phẩm nào "
    "một khách hàng thật sự có ý định mua trong giai đoạn sắp tới, chứ không chỉ là "
    "sản phẩm mà khách hàng tình cờ xem qua.",

    "Xuất phát từ yêu cầu đó, đề án nghiên cứu và xây dựng mô hình dự báo hành vi "
    "khách hàng dựa trên đồ thị tri thức động. Điểm khác biệt so với các mô hình gợi ý "
    "thông thường nằm ở chỗ mỗi tương tác giữa khách hàng và sản phẩm không được xem "
    "như nhau: một lượt thêm vào giỏ hàng có giá trị hơn một lượt xem, và một hành vi "
    "vừa diễn ra tuần trước có giá trị hơn hành vi từ nhiều tháng trước. Hai yếu tố "
    "hành vi và thời gian được đưa trực tiếp vào trọng số cạnh của đồ thị. Trên nền đồ "
    "thị đó, mô hình BT-DKGRec-GCN học biểu diễn của khách hàng và sản phẩm, rồi đưa "
    "ra danh sách những sản phẩm mà khách hàng có nhiều khả năng mua nhất.",

    "Mô hình được thực nghiệm trên bộ dữ liệu Retail Rocket, ghi lại các lượt xem, "
    "thêm vào giỏ và mua hàng thực tế của một sàn thương mại điện tử. Ở chỉ số chính, "
    "đo tỷ lệ sản phẩm khách hàng thật sự mua được mô hình đưa vào danh sách hai mươi "
    "gợi ý, mô hình đề xuất vượt LightGCN 28% trên nhóm khách hàng chung và 33% trên "
    "nhóm khách hàng có nhiều tương tác hơn; LightGCN là mô hình gợi ý trên đồ thị "
    "đang được sử dụng phổ biến. Mô hình đề xuất đứng đầu trên cả bốn chỉ số đánh giá "
    "và ở cả hai nhóm khách hàng, và lần chạy cho kết quả thấp nhất của mô hình vẫn "
    "cao hơn lần chạy tốt nhất của LightGCN.",

    "Sản phẩm ứng dụng của đề tài là một chương trình web cho phép chọn từng khách "
    "hàng cụ thể để xem lịch sử hành vi, danh sách sản phẩm được mô hình gợi ý và lý "
    "do đằng sau mỗi gợi ý: sản phẩm đó được đề xuất vì khách hàng đã từng tương tác "
    "trực tiếp, hay vì nó cùng danh mục với những sản phẩm khách hàng đã xem.",

    "Từ khoá: đồ thị tri thức động; dự báo hành vi khách hàng; hệ thống gợi ý; "
    "mạng nơ-ron đồ thị; Retail Rocket.",
]

SUMMARY = [
    "On e-commerce platforms, each customer reaches only a small fraction of the tens "
    "of thousands of items on offer, and most views never lead to a purchase. The "
    "problem this poses for the business is to predict which items a customer "
    "genuinely intends to buy in the coming period, rather than which items the "
    "customer merely happened to look at.",

    "Motivated by this need, the project studies and builds a customer behavior "
    "prediction model based on a dynamic knowledge graph. What distinguishes it from "
    "conventional recommendation models is that not every customer-item interaction "
    "counts equally: adding an item to the cart carries more weight than viewing it, "
    "and an action taken last week carries more weight than one from months earlier. "
    "Both factors, the type of behavior and its recency, are written directly into the "
    "edge weights of the graph. On that graph, the BT-DKGRec-GCN model learns "
    "representations of customers and items and produces a ranked list of the items "
    "each customer is most likely to buy.",

    "The model was evaluated on the Retail Rocket dataset, which records real views, "
    "cart additions, and purchases from an e-commerce site. On the primary metric, the "
    "share of items a customer actually bought that the model placed in a list of "
    "twenty recommendations, the proposed model outperforms LightGCN - a graph-based "
    "recommender in widespread use - by 28% on the general customer group and 33% on "
    "the group with denser interaction histories. The proposed model ranks first on "
    "all four evaluation metrics and in both customer groups, and its weakest run "
    "still exceeds the strongest run of LightGCN.",

    "The applied outcome of the project is a web program that allows an individual "
    "customer to be selected in order to inspect that customer's behavior history, the "
    "items the model recommends, and the reason behind each recommendation: whether "
    "the item is suggested because the customer interacted with it directly, or "
    "because it shares a category with items the customer has viewed.",

    "Keywords: dynamic knowledge graph; customer behavior prediction; recommender "
    "system; graph neural network; Retail Rocket.",
]


# ── Doc so lieu ─────────────────────────────────────────────────────────


def doc_tung_seed(runs: Path) -> dict[tuple[str, str], dict[int, float]]:
    """Recall@20 tren tap test, nhom warm, theo (cohort, model, seed).

    Doc thang tu metrics.json de khong bao gio go tay con so vao bang.
    """
    out: dict[tuple[str, str], dict[int, float]] = {}
    files = sorted(glob.glob(str(runs / "*" / "metrics.json")))
    if not files:
        raise SystemExit(f"LOI: khong thay metrics.json nao trong {runs}")
    for path in files:
        m = json.loads(Path(path).read_text())
        out.setdefault((m["cohort"], m["model"]), {})[m["seed"]] = (
            m["test"]["warm"]["recall@20"])
    thieu = [k for k, v in out.items() if sorted(v) != SEEDS]
    if thieu:
        raise SystemExit(f"LOI: thieu seed o {thieu}")
    return out


def kiem_tra_khong_chong_lan(so: dict) -> None:
    """Khang dinh trong Tom tat phai dung TRUOC khi ghi no vao tai lieu.

    Neu mot lan chay nao do lam LightGCN vuot len, cau van do thanh sai va
    script phai dung, chu khong duoc lang le xuat ban.
    """
    for cohort, _ in COHORTS:
        lgn = max(so[(cohort, "lightgcn")].values())
        bt = min(so[(cohort, "bt_dkgrec_l05")].values())
        if bt <= lgn:
            raise SystemExit(
                f"LOI: tren cohort {cohort}, lan chay thap nhat cua BT-DKGRec "
                f"({bt:.6f}) KHONG cao hon lan chay tot nhat cua LightGCN "
                f"({lgn:.6f}). Phai sua lai cau trong Tom tat truoc khi chay.")
        log.info("%s: BT thap nhat %.6f > LightGCN cao nhat %.6f", cohort, bt, lgn)


# ── Tien ich tai lieu ───────────────────────────────────────────────────


def para_khop(doc: Document, text: str):
    """Caption trong THAN BAI. Loai bo List Bullet vi danh muc bang trich lai
    nguyen van cac caption -- do la viec cua no, khong phai ban trung."""
    hits = [p for p in doc.paragraphs
            if p.text.strip() == text and p.style.name != "List Bullet"]
    if len(hits) != 1:
        raise SystemExit(f"LOI: doan {text[:60]!r} khop {len(hits)} lan, can dung 1.")
    return hits[0]


def heading_para(doc: Document, text: str):
    hits = [p for p in doc.paragraphs
            if p.text.strip() == text and p.style.name.startswith("Heading")]
    if len(hits) != 1:
        raise SystemExit(f"LOI: tieu de {text!r} khop {len(hits)} lan, can dung 1.")
    return hits[0]


def dat_text(para, text: str) -> None:
    """Thay noi dung, giu lai dinh dang cua run dau tien."""
    if not para.runs:
        para.add_run(text)
    else:
        para.runs[0].text = text
        for run in para.runs[1:]:
            run._r.getparent().remove(run._r)
    TOUCHED.append(para)


def caption_para(doc: Document, text: str):
    """Caption bang: Normal, can giua, in dam -- dung quy uoc cua 08_make_docx.py."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(text).bold = True
    return p


def bang_moi(doc: Document, header: list[str], rows: list[list[str]]):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for cell, text in zip(t.rows[0].cells, header):
        cell.text = ""
        cell.paragraphs[0].add_run(text).bold = True
    for row in rows:
        cells = t.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = str(text)
    return t


# ── Cac buoc va ─────────────────────────────────────────────────────────


def chen_bang_47(doc: Document, so: dict) -> None:
    """Chen Bang 4.7 vao cuoi muc 4.2.3, ngay truoc tieu de 4.2.4."""
    moc = heading_para(doc, "4.2.4. Kết quả thí nghiệm loại bỏ thành phần")

    header = ["Nhóm người dùng", "Mô hình"] + [f"seed {s}" for s in SEEDS]
    rows = []
    for cohort, nhan_cohort in COHORTS:
        for model, nhan_model in MODEL_ORDER:
            gia_tri = so[(cohort, model)]
            rows.append([nhan_cohort, nhan_model]
                        + [f"{gia_tri[s]:.6f}" for s in SEEDS])

    dan = doc.add_paragraph(
        "Bảng 4.5 và Bảng 4.6 trình bày giá trị trung bình trên ba seed. Bảng 4.7 "
        "dưới đây liệt kê kết quả của từng lần chạy, để có thể đối chiếu trực tiếp "
        "mức dao động giữa các seed với khoảng cách giữa các mô hình.")
    cap = caption_para(doc, CAPTION_47)
    tbl = bang_moi(doc, header, rows)
    trong = doc.add_paragraph()

    lgn_o = max(so[("original", "lightgcn")].values())
    bt_o = min(so[("original", "bt_dkgrec_l05")].values())
    lgn_a = max(so[("active", "lightgcn")].values())
    bt_a = min(so[("active", "bt_dkgrec_l05")].values())
    nhan_xet = doc.add_paragraph(
        f"Trên nhóm người dùng có lịch sử ban đầu, lần chạy cho kết quả cao nhất của "
        f"LightGCN đạt {lgn_o:.6f}, trong khi lần chạy cho kết quả thấp nhất của "
        f"BT-DKGRec-GCN với λ=0.05 đạt {bt_o:.6f}. Trên nhóm người dùng tích cực, hai "
        f"giá trị tương ứng là {lgn_a:.6f} và {bt_a:.6f}. Ở cả hai nhóm, ba lần chạy "
        f"của mô hình đề xuất đều nằm trên ba lần chạy của LightGCN, không có lần nào "
        f"chồng lấn. Popularity và Recent Popularity là mô hình tất định nên ba lần "
        f"chạy cho cùng một giá trị.")

    # Chen theo dung thu tu: moi phan tu duoc dat ngay truoc tieu de 4.2.4.
    for el in (dan._p, cap._p, tbl._tbl, trong._p, nhan_xet._p):
        moc._p.addprevious(el)

    TOUCHED.extend([dan, cap, tbl, nhan_xet])
    CHANGELOG.append((
        "Chương 4, mục 4.2.3",
        "Thêm Bảng 4.7 liệt kê Recall@20 của từng lần chạy trên ba seed",
        "Tóm tắt khẳng định lần chạy thấp nhất của mô hình đề xuất vẫn cao hơn lần "
        "chạy tốt nhất của LightGCN; v15 không có số liệu từng seed để chứng minh"))


def danh_so_lai(doc: Document) -> None:
    for cu, moi in DOI_SO.items():
        dat_text(para_khop(doc, cu), moi)
    CHANGELOG.append((
        "Chương 4, đánh số bảng",
        "Bảng 4.4 → 4.3, Bảng 4.4b → 4.4, Bảng 4.8 (mục 4.5.5) → 4.13",
        "v15 thiếu Bảng 4.3 và 4.7, có số hiệu chắp vá 4.4b, và dùng số 4.8 hai lần"))


def viet_lai_danh_muc_bang(doc: Document) -> None:
    """Danh muc bang Chuong 4 cua v15 dung o 4.8 trong khi than bai co toi 4.12."""
    cu = [p for p in doc.paragraphs
          if p.style.name == "List Bullet" and p.text.strip().startswith("Bảng 4.")]
    if not cu:
        raise SystemExit("LOI: khong thay muc nao cua Chuong 4 trong danh muc bang.")

    moc = cu[0]._p
    for text in DANH_MUC_C4:
        p = doc.add_paragraph(text, style="List Bullet")
        moc.addprevious(p._p)
        TOUCHED.append(p)
    for p in cu:
        p._p.getparent().remove(p._p)

    CHANGELOG.append((
        "Danh mục bảng",
        f"Viết lại {len(DANH_MUC_C4)} mục của Chương 4 theo đúng thân bài",
        f"Danh mục cũ chỉ có {len(cu)} mục, dừng ở Bảng 4.8, và ghi sai tên nhiều bảng"))


def viet_lai_tom_tat(doc: Document) -> None:
    for tieu_de, noi_dung, ten in (
            ("TÓM TẮT ĐỀ ÁN THẠC SĨ", TOM_TAT, "Tóm tắt"),
            ("SUMMARY OF MASTER'S PROJECT", SUMMARY, "Summary")):
        head = heading_para(doc, tieu_de)
        cu, el = [], head._p.getnext()
        while el is not None:
            if el.tag == qn("w:p"):
                style = el.find(f".//{qn('w:pStyle')}")
                name = style.get(qn("w:val")) if style is not None else None
                if name and name.startswith("Heading"):
                    break
            cu.append(el)
            el = el.getnext()

        neo = head._p
        for text in noi_dung:
            p = doc.add_paragraph(text)
            neo.addnext(p._p)
            neo = p._p          # moi doan noi tiep doan vua chen
            TOUCHED.append(p)
        for el in cu:
            el.getparent().remove(el)

        CHANGELOG.append((
            ten,
            "Viết lại toàn bộ, bỏ các giá trị Recall/NDCG/HitRate chi tiết",
            "Bản cũ có 25 con số và 8 giá trị thập phân bốn chữ số; hai bài mẫu của "
            "trường có 3 và 0. Con số chi tiết vẫn nằm đủ ở Chương 4"))


def append_changelog(doc: Document) -> None:
    tables = [t for t in doc.tables
              if t.rows and "Mục" in t.rows[0].cells[0].text]
    if len(tables) != 1:
        log.warning("Khong thay bang danh muc sua doi (%d bang khop); bo qua.",
                    len(tables))
        return
    for muc, sua, ly_do in CHANGELOG:
        cells = tables[0].add_row().cells
        for cell, text in zip(cells, (f"[v16] {muc}", sua, ly_do)):
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
    ap.add_argument("--out", type=Path, default=Path("docs/De_an_thac_si_v16.docx"))
    ap.add_argument("--no-highlight", action="store_true")
    args = ap.parse_args()

    if not args.source.exists():
        raise SystemExit(f"LOI: khong thay {args.source}")

    so = doc_tung_seed(args.runs)
    kiem_tra_khong_chong_lan(so)

    doc = Document(str(args.source))
    ct_truoc = len(doc.element.body.findall(f".//{OMML}oMath"))
    bang_truoc, hinh_truoc = len(doc.tables), len(doc.inline_shapes)

    chen_bang_47(doc, so)
    danh_so_lai(doc)
    viet_lai_danh_muc_bang(doc)
    viet_lai_tom_tat(doc)
    append_changelog(doc)

    ct_sau = len(doc.element.body.findall(f".//{OMML}oMath"))
    if ct_sau != ct_truoc:
        raise SystemExit(f"LOI: cong thuc {ct_truoc} -> {ct_sau}; khong duoc mat cai nao.")
    if len(doc.inline_shapes) != hinh_truoc:
        raise SystemExit(f"LOI: hinh {hinh_truoc} -> {len(doc.inline_shapes)}.")
    if len(doc.tables) != bang_truoc + 1:
        raise SystemExit(f"LOI: bang {bang_truoc} -> {len(doc.tables)}, can +1.")

    log.info("cong thuc %d · bang %d (truoc %d) · hinh %d",
             ct_sau, len(doc.tables), bang_truoc, len(doc.inline_shapes))

    if args.no_highlight:
        log.info("Xoa to nen o %d run", xoa_to_nen(doc))
    else:
        log.info("To nen vang %d run", highlight_touched())

    doc.save(str(args.out))
    log.info("Da ghi %s", args.out)


if __name__ == "__main__":
    main()
