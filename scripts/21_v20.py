"""v19 -> v20: ban trinh bay cho co.

Hinh 3.1 ve he thong theo thiet ke muc 3.7.1, net dut la phan ban hien tai chua noi.
Ve lai sau hinh Chuong 4 (scripts/22).
Sua cau ve hoi tu cho dung voi curves.csv, dong bo danh muc hinh/bang voi chu thich.

Chay:
    python scripts/21_v20.py
    python scripts/21_v20.py --no-highlight --bo-changelog --out docs/De_an_thac_si_v20_ban_nop.docx

Ghi fig_v20_3_1_kien_truc.{drawio,png}. Chay lai se ghi de cho da chinh tay trong draw.io.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "De_an_thac_si_v19.docx"
RUNS = ROOT / "experiments" / "runs"
HINH_3_1 = ROOT / "docs" / "figures" / "fig_v20_3_1_kien_truc.png"
MO_HINH_DO_THI = ("lightgcn", "static_kg_gcn", "bt_dkgrec", "bt_dkgrec_l05")
TRAN_EPOCH = 1000


def _nap(ten: str, file: str):
    spec = importlib.util.spec_from_file_location(ten, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[ten] = mod
    spec.loader.exec_module(mod)
    return mod


v18 = _nap("v18", "17_v18.py")
dio = _nap("dio", "19_drawio_chuong3.py")
bd = _nap("bd", "22_bieu_do_chuong4.py")

from docx import Document  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402

CHANGELOG = [
    ("Hình 3.1", "Vẽ theo thiết kế mục 3.7.1; nét đứt là phần bản hiện tại chưa nối.",
     "Tên hình là kiến trúc tổng quan của hệ thống, hình cũ chỉ tả bản demo."),
    ("Mục 4.2.1 và 5.3.1", "Sửa khoảng epoch tốt nhất và câu về hội tụ, đọc từ curves.csv.",
     "Nhóm Active phần lớn chạm trần 1000 epoch; một lần chạy λ=0.05 trên Original tốt nhất ở epoch 400."),
    ("Hình 4.1–4.6", "Vẽ lại đúng khổ in, font Times, chữ nằm ngang; số đọc lại từ experiments/runs.",
     "Bản cũ có vân sọc, số ghi nhiều kiểu, nhãn tiếng Anh, chữ đè nhau."),
    ("Danh mục hình và bảng", "Đồng bộ với chú thích.", "Sáu dòng gõ tay lệch với chú thích dưới hình, trên bảng."),
]


def doc_hoi_tu(runs: Path) -> dict:
    """Doc diem dung va epoch tot nhat cua cac mo hinh do thi tu curves.csv."""
    so = {}
    for cohort in ("original", "active"):
        dong = []
        for model in MO_HINH_DO_THI:
            files = sorted(runs.glob(f"{cohort}_{model}_20[0-9][0-9]_*/curves.csv"))
            if len(files) != 3:
                raise SystemExit(f"LOI: {cohort}/{model} co {len(files)} run, can 3")
            for f in files:
                rows = list(csv.DictReader(f.open(encoding="utf-8")))
                danh_gia = [r for r in rows if r.get("valid_ndcg@20")]
                tot = max(danh_gia, key=lambda r: float(r["valid_ndcg@20"]))
                dong.append((model, int(rows[-1]["epoch"]), int(tot["epoch"])))
        so[cohort] = dong
    return so


def cau_hoi_tu(so: dict) -> str:
    original, active = so["original"], so["active"]
    if any(cuoi >= TRAN_EPOCH for _, cuoi, _ in original):
        raise SystemExit("LOI: co run Original cham tran, phai viet lai cau hoi tu")
    cham = [tot for _, cuoi, tot in active if cuoi >= TRAN_EPOCH]
    return (
        "Mỗi cấu hình được chạy ba seed với cơ chế dừng sớm theo tập xác thực. Trên nhóm Original, "
        f"cả {len(original)} lần chạy của bốn mô hình đồ thị đều dừng sớm. Trên nhóm Active, "
        f"{len(cham)} trong {len(active)} lần chạy chạm trần {TRAN_EPOCH} epoch, với epoch tốt nhất "
        f"từ {min(cham)} đến {max(cham)}, nên kết quả của nhóm này có thể chưa đạt mức hội tụ cuối cùng."
    )


def khoang_original(so: dict) -> tuple[str, str]:
    """Cum cu va cum moi ve khoang epoch tot nhat tren Original."""
    tot = sorted(t for _, _, t in so["original"])
    if tot[0] != 50 or tot[-2] != 145 or tot[-1] != 400:
        raise SystemExit(f"LOI: khoang epoch Original doi: {tot}")
    return ("trong khoảng epoch 50 đến 145, tức gấp năm đến mười lăm lần ngân sách của v11",
            "trong khoảng epoch 50 đến 145, riêng một lần chạy của mô hình λ=0.05 ở epoch 400, "
            "tức gấp ít nhất năm lần ngân sách của v11")


def dong_bo_danh_muc(doc) -> int:
    chu_thich, danh_muc = {}, {}
    for p in doc.paragraphs:
        m = re.match(r"^((?:Hình|Bảng) \d+\.\d+)\.", p.text.strip())
        if m:
            (danh_muc if p.style.name == "List Bullet" else chu_thich)[m.group(1)] = p
    if set(chu_thich) != set(danh_muc):
        raise SystemExit(f"LOI: danh muc lech chu thich: {set(chu_thich) ^ set(danh_muc)}")
    n = 0
    for ten, p in danh_muc.items():
        if p.text.strip() != chu_thich[ten].text.strip():
            v18.dat_text(p, chu_thich[ten].text.strip())
            n += 1
    return n


def hinh_3_1(da_noi: bool = False) -> str:
    """da_noi: demo da chay theo thiet ke, chi con nhanh anh chup do thi la net dut."""
    g = dio.Do()
    ink, muted, accent = dio.INK, dio.MUTED, dio.ACCENT

    def nhan(ten, *dong, color=muted):
        s = f'<b style="font-size:17px">{ten}</b>'
        for d in dong:
            s += f'<br><font style="font-size:14.5px" color="{color}">{d}</font>'
        return s

    box = f"rounded=1;arcSize=5;whiteSpace=wrap;html=1;strokeColor={ink};strokeWidth=1;fillColor=#FFFFFF;spacing=4;"
    lan = "rounded=1;arcSize=3;html=1;strokeColor=#C9CDD3;strokeWidth=1;fillColor=#F6F7F8;"
    lien = f"rounded=1;strokeWidth=1.3;strokeColor={ink};"
    dut = f"rounded=1;strokeWidth=1.2;strokeColor={muted};dashed=1;dashPattern=5 4;"
    moi = lien if da_noi else dut
    x = [0, 216, 432, 648]
    w = 164

    g.node(-12, 98, 836, 122, "", lan, "lan1")
    g.node(-12, 328, 836, 122, "", lan, "lan2")
    tieu_de = "fontStyle=1;spacingLeft=2;"
    g.text(-10, 74, 420, 22, "CẬP NHẬT ĐỒ THỊ CẬN THỜI GIAN THỰC", size=12.5, align="left", color=muted, extra=tieu_de)
    g.text(-10, 452, 420, 22, "HUẤN LUYỆN NGOẠI TUYẾN THEO CHU KỲ", size=12.5, align="left", color=muted, extra=tieu_de)

    rr = g.node(x[3], 0, w, 60, nhan("Dữ liệu<br>Retail Rocket"), box + "fillColor=#F6F7F8;", "rr")
    app = g.node(x[0], 110, w, 96, nhan("Ứng dụng", "khách xem, thêm giỏ,", "mua và nhận gợi ý"), box, "app")
    tn = g.node(x[1], 110, w, 96, nhan("Tiếp nhận sự kiện", "xem, thêm giỏ, mua"), box, "tn")
    cn = g.node(x[2], 110, w, 96, nhan("Cập nhật đồ thị", "W(u,i) = Σα<sub>b</sub>e<sup>−λΔt</sup>", color=accent),
                box + f"strokeColor={accent};strokeWidth=1.6;fillColor=#FCF1F0;", "cn")
    kho = g.node(x[3], 110, w, 96, nhan("Kho đồ thị", "Neo4j"),
                 f"shape=cylinder3;boundedLbl=1;size=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor={ink};strokeWidth=1;",
                 "kho")
    top = g.node(x[0], 340, w, 96, nhan("Dịch vụ dự báo", "Top-K cho từng khách"), box, "top")
    hl = g.node(x[3], 340, w, 96, nhan("Huấn luyện", "BT-DKGRec-GCN", "GCN 3 lớp, BPR"), box, "hl")

    g.edge(rr, kho, "", lien + "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    g.text(x[3] - 92, 70, 170, 20, "nạp ban đầu", size=14, align="right")
    g.edge(rr, hl, "", lien + "exitX=1;exitY=0.5;entryX=1;entryY=0.5;", points=[(836, 30), (836, 388)])
    g.edge(app, tn, "", moi + "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    g.edge(tn, cn, "", moi + "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    g.edge(cn, kho, "", moi + "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    g.edge(kho, app, "", moi + "exitX=0.15;exitY=1;entryX=0.75;entryY=1;", points=[(x[3] + 24.6, 250), (x[0] + 123, 250)])
    g.text(340, 227, 200, 22, "lịch sử hành vi", size=14)
    g.edge(kho, top, "", moi + "exitX=0.4;exitY=1;entryX=0.75;entryY=0;", points=[(x[3] + 65.6, 296), (x[0] + 123, 296)])
    g.text(340, 273, 200, 22, "lịch sử của khách" if da_noi else "trọng số cạnh của khách", size=14)
    g.edge(kho, hl, "", dut + "exitX=0.72;exitY=1;entryX=0.72;entryY=0;")
    g.text(x[3] + 124, 258, 76, 40, "ảnh chụp<br>đồ thị", size=14, align="left")
    g.edge(hl, top, "", lien + "exitX=0;exitY=0.5;entryX=1;entryY=0.5;")
    g.text(340, 364, 200, 22, "mô hình đã huấn luyện" if da_noi else "kết quả huấn luyện", size=14)
    g.edge(top, app, "", lien + "exitX=0.25;exitY=0;entryX=0.25;entryY=1;")
    g.text(-40, 256, 76, 40, "danh sách<br>Top-K", size=14, align="right")
    g.text(-10, 478, 850, 42,
           ("<i>Nét liền: đã hiện thực. Nét đứt: theo thiết kế, huấn luyện lại theo chu kỳ chưa tự động.<br>" if da_noi else
            "<i>Nét liền: đã hiện thực. Nét đứt: theo thiết kế, bản hiện tại chưa nối.<br>")
           +
           "Màu đỏ: phần đóng góp của đề án (trọng số cạnh). Kiến trúc lan truyền giữ nguyên LightGCN.</i>", size=14)
    return g.xml("Hinh 3.1")


def thay_anh(doc, caption: str, path: Path) -> None:
    """Thay anh ngay truoc chu thich, giu chieu rong, chieu cao theo ti le anh moi."""
    from PIL import Image

    cap, blip = v18._blip_truoc_caption(doc, caption)
    rid, _ = doc.part.get_or_add_image(io.BytesIO(path.read_bytes()))
    blip.set(qn("r:embed"), rid)
    w, h = Image.open(path).size
    inline = blip
    while inline is not None and inline.tag not in (qn("wp:inline"), qn("wp:anchor")):
        inline = inline.getparent()
    for ext in [inline.find(qn("wp:extent")), *inline.iter(qn("a:ext"))]:
        if ext is not None and ext.get("cx"):
            ext.set("cy", str(int(int(ext.get("cx")) * h / w)))
    v18.TOUCHED.append(cap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "De_an_thac_si_v20.docx")
    ap.add_argument("--no-highlight", action="store_true")
    ap.add_argument("--bo-changelog", action="store_true", help="go trang danh muc sua doi -- ban gui co")
    args = ap.parse_args()

    v18.TOUCHED.clear()
    so = doc_hoi_tu(args.runs)
    doc = Document(str(args.source))
    dem = (len(doc.element.body.findall(f".//{v18.OMML}oMath")), len(doc.tables), len(doc.inline_shapes))

    v18.xoa_to_nen(doc)
    v18.bo_cum(doc, "Mỗi cấu hình được chạy ba seed và huấn luyện đến khi",
               "Mỗi cấu hình được chạy ba seed và huấn luyện đến khi kết quả trên tập xác thực không còn cải thiện.",
               cau_hoi_tu(so))
    cu, moi = khoang_original(so)
    v18.bo_cum(doc, "Thứ nhất, mô hình được huấn luyện đến khi validation", cu, moi)
    print(f"dong bo danh muc: {dong_bo_danh_muc(doc)} dong")
    dio.render(hinh_3_1(), HINH_3_1, 930, 560)
    thay_anh(doc, "Hình 3.1.", HINH_3_1)
    for caption, path in bd.main(args.runs).items():
        thay_anh(doc, caption, path)

    bang = [t for t in doc.tables if t.rows and "Mục" in t.rows[0].cells[0].text]
    if len(bang) != 1:
        raise SystemExit(f"LOI: bang danh muc sua doi khop {len(bang)} lan, can 1")
    for muc, sua, ly_do in CHANGELOG:
        cells = bang[0].add_row().cells
        for cell, text in zip(cells, (f"[v20] {muc}", sua, ly_do)):
            cell.text = text
            v18.TOUCHED.extend(cell.paragraphs)

    if v18.hinh_hong(doc):
        raise SystemExit(f"LOI: con hinh hong {v18.hinh_hong(doc)}")
    if (len(doc.element.body.findall(f".//{v18.OMML}oMath")), len(doc.tables), len(doc.inline_shapes)) != dem:
        raise SystemExit("LOI: so cong thuc / bang / hinh bi doi")

    if args.bo_changelog:
        print(f"go trang danh muc sua doi: {v18.xoa_muc_changelog(doc)} phan tu")
    if args.no_highlight:
        v18.xoa_to_nen(doc)
    else:
        print(f"to nen vang {v18.to_nen()} run")
    doc.save(str(args.out))
    print(f"cong thuc {dem[0]} · bang {len(doc.tables)} · hinh {dem[2]} -> {args.out}")


if __name__ == "__main__":
    main()
