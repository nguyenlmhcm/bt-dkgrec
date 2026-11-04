"""Va v17 -> v18: sua hai muc han che con chu cua v11, dua ket qua D35 vao muc 4.2.4.

Ly do ton tai:

1. Muc 4.4.4 va 5.3.1 van chep nguyen van ban v11: "mot seed (2026)", "10 epoch",
   "chua co do lech chuan". Trong khi muc 4.2.1 cua CHINH quyen nay viet "ba seed
   thay vi mot" va "huan luyen den khi validation khong con cai thien thay vi dung
   o 10 epoch". Quyen tu mau thuan -- loi nang nhat, sua truoc khi gui co.

2. Muc 4.2.4 so mo hinh de xuat voi Static KG-GCN bang so cua lambda=0.01
   (0.028778), trong khi mo hinh de xuat da chot la lambda=0.05 (Tom tat,
   Bang 4.8, v17).

3. Muc 4.2.4 noi tach alpha khoi lambda "la huong phat trien" -- D35 da lam xong
   (docs/DECISIONS.md). Ket qua viet bang loi kem so, KHONG them bang, de khong
   phai danh so lai bang Chuong 4.

4. Bo chu "cuc bo" trong cum "ung dung nguyen mau cuc bo" (Mo dau, muc 5.1.3).

5. Sau HINH 4.1-4.6 TRONG TU v15: buoc 13 chep khung hinh (a:blip) tu
   Chuong4_viet_lai.docx sang nhung KHONG chep phan anh, nen rId21-rId26 tro vao
   khong khi. Gan lai anh goc tu chinh file do.

6. Ve lai Hinh 3.1-3.3 (scripts/18_ve_hinh_chuong3.py): anh cu tu v11 ghi
   "GCN 2 lop", "BPR co trong so", co node "metadata" khong ton tai, nhan tieng Anh.

Moi con so doc tu experiments/runs/ -- khong go tay.

Chay:
    python scripts/17_v18.py                                   # ban ra soat
    python scripts/17_v18.py --no-highlight --bo-changelog \\
        --out docs/De_an_thac_si_v18_ban_nop.docx              # ban gui co
"""

from __future__ import annotations

import argparse
import copy
import io
import glob
import json
import statistics as st
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "De_an_thac_si_v17.docx"
RUNS = ROOT / "experiments" / "runs"
OMML = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
CHANGELOG_HEAD = "DANH MỤC SỬA ĐỔI SO VỚI BẢN v11"
CHUONG4 = ROOT / "docs" / "Chuong4_viet_lai.docx"
BLIP = qn("a:blip")
EMBED = qn("r:embed")

TOUCHED: list = []


# ── So lieu ─────────────────────────────────────────────────────────────


def doc_so(runs: Path) -> dict:
    """Recall@20 trung binh, so seed thang Static, va so user theo bac."""
    rec: dict = {}
    users: dict = {}
    for f in glob.glob(str(runs / "*" / "metrics.json")):
        m = json.loads(Path(f).read_text())
        rec.setdefault((m["cohort"], m["model"]), {})[m["seed"]] = m["test"]["warm"]["recall@20"]
        users[m["cohort"]] = m["test"]["n_users"]
    can = ["static_kg_gcn", "bt_dkgrec_alpha_only", "bt_dkgrec_time_only", "bt_dkgrec_l05"]
    for cohort in ("original", "active"):
        for model in can:
            if sorted(rec.get((cohort, model), {})) != [2020, 2021, 2022]:
                raise SystemExit(f"LOI: {cohort}/{model} chua du 3 seed")
    so = {"mean": {k: st.mean(v.values()) for k, v in rec.items()}, "users": users}
    static = rec[("active", "static_kg_gcn")]
    so["time_thang_active"] = sum(rec[("active", "bt_dkgrec_time_only")][s] > static[s] for s in static)
    return so


def f6(x: float) -> str:
    return f"{x:.6f}"


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%".replace(".", ",")


# ── Tien ich ────────────────────────────────────────────────────────────


def para_bat_dau(doc: Document, prefix: str):
    hits = [p for p in doc.paragraphs
            if p.text.strip().startswith(prefix) and p.style.name != "List Bullet"]
    if len(hits) != 1:
        raise SystemExit(f"LOI: doan bat dau {prefix[:60]!r} khop {len(hits)} lan, can 1.")
    return hits[0]


def dat_text(para, text: str) -> None:
    if not para.runs:
        para.add_run(text)
    else:
        para.runs[0].text = text
        for run in para.runs[1:]:
            run._r.getparent().remove(run._r)
    TOUCHED.append(para)


def chen_sau(para, text: str):
    """Them mot doan cung dinh dang ngay sau `para`."""
    moi = copy.deepcopy(para._p)
    para._p.addnext(moi)
    p = Paragraph(moi, para._parent)
    dat_text(p, text)
    return p


def bo_cum(doc: Document, prefix: str, cum: str, thay: str) -> None:
    para = para_bat_dau(doc, prefix)
    if cum not in para.text:
        raise SystemExit(f"LOI: khong thay {cum!r} trong doan {prefix[:40]!r}")
    dat_text(para, para.text.replace(cum, thay))


# ── Cac ban va ──────────────────────────────────────────────────────────


def va_424(doc: Document, so: dict) -> None:
    m = so["mean"]
    p = para_bat_dau(doc, "Bỏ trọng số hành vi-thời gian khỏi mô hình đề xuất thì thu được đúng Static KG-GCN")
    dau = p.text.split("khác nhau duy nhất ở hàm tính trọng số cạnh.")[0]
    dat_text(p, dau + "khác nhau duy nhất ở hàm tính trọng số cạnh. Với mô hình đề xuất (λ=0.05), "
             f"Recall@20 giảm từ {f6(m[('original', 'bt_dkgrec_l05')])} xuống "
             f"{f6(m[('original', 'static_kg_gcn')])} trên nhóm người dùng có lịch sử ban đầu khi bỏ "
             f"thành phần này; trên nhóm người dùng tích cực, giá trị giảm từ "
             f"{f6(m[('active', 'bt_dkgrec_l05')])} xuống {f6(m[('active', 'static_kg_gcn')])}.")

    p = para_bat_dau(doc, "Hệ số cường độ hành vi α được đặt cố định ở ba mức")
    dat_text(p, "Để tách đóng góp của hệ số cường độ hành vi α khỏi hệ số suy giảm λ, đề án chạy "
             "bổ sung hai cấu hình, mỗi cấu hình khác mô hình đề xuất đúng một tham số: chỉ giữ α "
             "(λ=0) và chỉ giữ suy giảm thời gian (α=1 cho mọi hành vi). Trên nhóm người dùng tích "
             "cực, cấu hình chỉ giữ suy giảm thời gian đạt Recall@20 "
             f"{f6(m[('active', 'bt_dkgrec_time_only')])}, tương đương mô hình đầy đủ "
             f"({f6(m[('active', 'bt_dkgrec_l05')])}) và cao hơn Static KG-GCN ở "
             f"{'cả ba' if so['time_thang_active'] == 3 else so['time_thang_active']} seed; cấu hình "
             f"chỉ giữ α đạt {f6(m[('active', 'bt_dkgrec_alpha_only')])}. Trên nhóm người dùng có "
             "lịch sử ban đầu, từng thành phần đứng riêng không tạo khác biệt rõ so với Static "
             f"KG-GCN ({f6(m[('original', 'bt_dkgrec_alpha_only')])} và "
             f"{f6(m[('original', 'bt_dkgrec_time_only')])} so với "
             f"{f6(m[('original', 'static_kg_gcn')])}); chỉ khi kết hợp cả hai mới đạt "
             f"{f6(m[('original', 'bt_dkgrec_l05')])}.")
    uo, ua = so["users"]["original"], so["users"]["active"]
    chen_sau(p, "Kết quả cho thấy độ mới của tương tác phát huy rõ nhất khi người dùng có đủ lịch "
             f"sử. Ở nhóm người dùng tích cực, {pct(ua['warm_deg3plus'], ua['warm'])} người dùng "
             "được đánh giá có từ ba tương tác trở lên trong tập huấn luyện; ở nhóm người dùng có "
             f"lịch sử ban đầu, {pct(uo['warm_deg1'], uo['warm'])} người dùng được đánh giá chỉ có "
             "một tương tác. Với người dùng chỉ có một tương tác, không có thứ tự cũ mới nào để "
             "khai thác; loại hành vi là thông tin phân biệt mức độ quan tâm, nên hai thành phần "
             "cần đi cùng nhau. Giá trị của α vẫn được đặt cố định ở 1.0, 2.0 và 3.0, chưa dò "
             "trên dữ liệu.")


def va_han_che(doc: Document) -> None:
    dat_text(para_bat_dau(doc, "Kết quả hiện dùng một seed"),
             "Mỗi cấu hình được chạy ba seed; với số lần chạy này, đề án trình bày kết quả của "
             "từng seed (Bảng 4.7) thay vì khẳng định ý nghĩa thống kê. Ở chỉ số NDCG@20, mô hình "
             "đề xuất chỉ vượt LightGCN ở hai trong ba seed trên mỗi nhóm. Chỉ hệ số suy giảm λ "
             "của mô hình đề xuất được dò trên tập xác thực; hệ số α, tốc độ học và kích thước lô "
             "được đặt cố định cho mọi mô hình. Nhóm người dùng tích cực có 234 người dùng được "
             "đánh giá nên kết quả trên nhóm này mang ý nghĩa thăm dò.")
    dat_text(para_bat_dau(doc, "Các kết quả hiện tại sử dụng một seed (2026)"),
             "Mỗi cấu hình được chạy ba seed và huấn luyện đến khi kết quả trên tập xác thực "
             "không còn cải thiện. Với ba seed, đề án trình bày kết quả theo từng lần chạy thay vì "
             "khẳng định ý nghĩa thống kê; dao động giữa các seed cùng bậc độ lớn với khoảng cách "
             "giữa một số cặp mô hình kề nhau. Ngân sách lựa chọn siêu tham số chưa hoàn toàn đối "
             "xứng: hệ số suy giảm của mô hình đề xuất được dò trên tập xác thực, trong khi các mô "
             "hình đối chứng dùng cấu hình theo công bố gốc.")


def kiem_ndcg_2_tren_3(runs: Path) -> None:
    """Cau 'NDCG@20 chi vuot LightGCN o hai trong ba seed tren moi nhom' phai dung."""
    nd: dict = {}
    for f in glob.glob(str(runs / "*" / "metrics.json")):
        m = json.loads(Path(f).read_text())
        nd.setdefault((m["cohort"], m["model"]), {})[m["seed"]] = m["test"]["warm"]["ndcg@20"]
    for cohort in ("original", "active"):
        a, b = nd[(cohort, "bt_dkgrec_l05")], nd[(cohort, "lightgcn")]
        thang = sum(a[s] > b[s] for s in a)
        if thang != 2:
            raise SystemExit(f"LOI: NDCG@20 {cohort} thang {thang}/3, cau trong 4.4.4 sai")


# ── Hinh ────────────────────────────────────────────────────────────────


def hinh_hong(doc: Document) -> list[str]:
    """rId cua moi a:blip tro vao khong khi."""
    rels = doc.part.rels
    return [b.get(EMBED) for b in doc.element.body.iter(BLIP) if b.get(EMBED) not in rels]


def gan_lai_hinh_chuong4(doc: Document) -> int:
    """Lay anh goc tu Chuong4_viet_lai.docx theo dung rId da chep sang."""
    nguon = Document(str(CHUONG4)).part.rels
    n = 0
    for blip in doc.element.body.iter(BLIP):
        rid = blip.get(EMBED)
        if rid in doc.part.rels:
            continue
        if rid not in nguon:
            raise SystemExit(f"LOI: {rid} khong co trong ca {CHUONG4.name}")
        moi, _ = doc.part.get_or_add_image(io.BytesIO(nguon[rid].target_part.blob))
        blip.set(EMBED, moi)
        n += 1
    return n


def _blip_truoc_caption(doc: Document, caption: str):
    caps = [p for p in doc.paragraphs
            if p.text.strip().startswith(caption) and p.style.name != "List Bullet"]
    if len(caps) != 1:
        raise SystemExit(f"LOI: caption {caption!r} khop {len(caps)} lan, can 1")
    el = caps[0]._p.getprevious()
    for _ in range(3):
        if el is None:
            break
        blips = list(el.iter(BLIP))
        if blips:
            return caps[0], blips[0]
        el = el.getprevious()
    raise SystemExit(f"LOI: khong thay hinh ngay truoc {caption!r}")


def thay_hinh_chuong3(doc: Document) -> None:
    import importlib.util
    from PIL import Image

    hinh = {}
    # 18: Hinh 3.1 (SVG); 19: Hinh 3.2, 3.3 (draw.io, nguoi dung chinh tay duoc)
    for ten in ("18_ve_hinh_chuong3.py", "19_drawio_chuong3.py"):
        spec = importlib.util.spec_from_file_location(ten[:-3], ROOT / "scripts" / ten)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        hinh.update(mod.main())
    if sorted(hinh) != ["Hình 3.1.", "Hình 3.2.", "Hình 3.3."]:
        raise SystemExit(f"LOI: thieu hinh Chuong 3: {sorted(hinh)}")

    for caption, path in hinh.items():
        cap, blip = _blip_truoc_caption(doc, caption)
        rid, _ = doc.part.get_or_add_image(io.BytesIO(path.read_bytes()))
        blip.set(EMBED, rid)
        # Giu chieu rong cu, chinh chieu cao theo ti le anh moi -- khong meo hinh.
        w, h = Image.open(path).size
        inline = blip
        while inline is not None and inline.tag not in (qn("wp:inline"), qn("wp:anchor")):
            inline = inline.getparent()
        for ext in [inline.find(qn("wp:extent")), *inline.iter(qn("a:ext"))]:
            if ext is not None and ext.get("cx"):
                ext.set("cy", str(int(int(ext.get("cx")) * h / w)))
        TOUCHED.append(cap)


CHANGELOG = [
    ("Mục 4.4.4 và 5.3.1", "Viết lại: ba seed, huấn luyện đến hội tụ, NDCG@20 thắng 2/3 seed, α chưa dò.",
     "Hai mục còn chép nguyên văn v11 (một seed 2026, 10 epoch), mâu thuẫn với mục 4.2.1."),
    ("Mục 4.2.4, đoạn bỏ trọng số hành vi-thời gian", "Đổi số so sánh sang λ=0.05.",
     "Mô hình đề xuất đã chốt λ=0.05 (Tóm tắt, Bảng 4.8); đoạn cũ dùng số của λ=0.01."),
    ("Mục 4.2.4, đoạn về α", "Thay câu 'hướng phát triển' bằng kết quả tách α khỏi λ, thêm một đoạn diễn giải.",
     "Thí nghiệm tách α/λ đã chạy đủ 12 lần (D35); viết bằng lời, không thêm bảng."),
    ("Mở đầu và mục 5.1.3", "Bỏ chữ 'cục bộ' trong 'ứng dụng nguyên mẫu cục bộ'.", "Theo góp ý của tác giả."),
    ("Hình 4.1–4.6", "Gắn lại ảnh.", "Từ v15 sáu hình này trống: bước ghép Chương 4 chép khung hình nhưng không chép ảnh."),
    ("Hình 3.1–3.3", "Vẽ lại, nhãn tiếng Việt.",
     "Ảnh cũ từ v11 ghi 'GCN 2 lớp', 'BPR có trọng số', có node 'metadata' không tồn tại."),
]


def append_changelog(doc: Document) -> None:
    tables = [t for t in doc.tables if t.rows and "Mục" in t.rows[0].cells[0].text]
    if len(tables) != 1:
        raise SystemExit(f"LOI: bang danh muc sua doi khop {len(tables)} lan, can 1")
    for muc, sua, ly_do in CHANGELOG:
        cells = tables[0].add_row().cells
        for cell, text in zip(cells, (f"[v18] {muc}", sua, ly_do)):
            cell.text = text
            TOUCHED.extend(cell.paragraphs)


def xoa_muc_changelog(doc: Document) -> int:
    """Go trang danh muc sua doi. DUNG ngay sau bang nhat ky: tiep theo la trang bia."""
    head = [p for p in doc.paragraphs if p.text.strip() == CHANGELOG_HEAD]
    if not head:
        return 0
    n, el, thay_bang = 0, head[0]._p, False
    while el is not None and not thay_bang:
        ke = el.getnext()
        if el.tag == qn("w:tbl"):
            if "Mục" not in Table(el, doc).rows[0].cells[0].text:
                break
            thay_bang = True
        el.getparent().remove(el)
        n += 1
        el = ke
    if not thay_bang:
        raise SystemExit("LOI: khong thay bang nhat ky sua doi de go.")
    return n


def tat_ca_doan(doc: Document):
    yield from doc.paragraphs
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                yield from cell.paragraphs


def xoa_to_nen(doc: Document) -> int:
    n = 0
    for para in tat_ca_doan(doc):
        for run in para.runs:
            if run.font.highlight_color is not None:
                run.font.highlight_color = None
                n += 1
    return n


def to_nen() -> int:
    n = 0
    for para in TOUCHED:
        for run in para.runs:
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "De_an_thac_si_v18.docx")
    ap.add_argument("--no-highlight", action="store_true")
    ap.add_argument("--bo-changelog", action="store_true", help="go trang danh muc sua doi -- ban gui co")
    args = ap.parse_args()

    so = doc_so(args.runs)
    kiem_ndcg_2_tren_3(args.runs)
    doc = Document(str(args.source))
    ct, bang, hinh = (len(doc.element.body.findall(f".//{OMML}oMath")), len(doc.tables),
                      len(doc.inline_shapes))

    # To nen cu cua v17 bi xoa truoc: ban v18 chi to dung cho vua sua.
    xoa_to_nen(doc)
    va_424(doc, so)
    va_han_che(doc)
    bo_cum(doc, "Đề án đã hiện thực ứng dụng nguyên mẫu cục bộ", "nguyên mẫu cục bộ", "nguyên mẫu")
    bo_cum(doc, "Thiết kế kiến trúc cập nhật đồ thị cận thời gian thực", "nguyên mẫu cục bộ", "nguyên mẫu")
    print(f"gan lai {gan_lai_hinh_chuong4(doc)} hinh Chuong 4")
    thay_hinh_chuong3(doc)
    append_changelog(doc)
    if hinh_hong(doc):
        raise SystemExit(f"LOI: con hinh hong {hinh_hong(doc)}")

    than_bai = [p.text for p in doc.paragraphs]
    for cu in ("một seed (2026)", "Kết quả hiện dùng một seed", "10 epoch chưa đủ",
               "nguyên mẫu cục bộ", "Việc tách riêng đóng góp của α khỏi đóng góp của λ"):
        if any(cu in t for t in than_bai):
            raise SystemExit(f"LOI: con sot {cu!r}")
    if (len(doc.element.body.findall(f".//{OMML}oMath")), len(doc.tables),
            len(doc.inline_shapes)) != (ct, bang, hinh):
        raise SystemExit("LOI: so cong thuc / bang / hinh bi doi")

    if args.bo_changelog:
        print(f"go trang danh muc sua doi: {xoa_muc_changelog(doc)} phan tu")
    if args.no_highlight:
        xoa_to_nen(doc)
    else:
        print(f"to nen vang {to_nen()} run")
    doc.save(str(args.out))
    print(f"cong thuc {ct} · bang {len(doc.tables)} · hinh {hinh} -> {args.out}")


if __name__ == "__main__":
    main()
