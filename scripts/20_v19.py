"""v18 -> v19: mo ta ung dung va Neo4j cho dung voi code; thong nhat ten hai do thi.

Ung dung doc thang bo tep CSV (src/export/trace.py) va topk.csv, khong ket noi Neo4j.
Ban v18 co hai cau va Hinh 3.1 noi nguoc lai, mau thuan voi muc mo ta buoc xuat du lieu.

Chay:
    python scripts/20_v19.py
    python scripts/20_v19.py --no-highlight --bo-changelog --out docs/De_an_thac_si_v19_ban_nop.docx

Ghi fig_v19_3_1_kien_truc.{drawio,png}. Chay lai se ghi de cho da chinh tay trong draw.io.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "De_an_thac_si_v18.docx"
HINH_3_1 = ROOT / "docs" / "figures" / "fig_v19_3_1_kien_truc.png"


def _nap(ten: str, file: str):
    spec = importlib.util.spec_from_file_location(ten, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[ten] = mod
    spec.loader.exec_module(mod)
    return mod


v18 = _nap("v18", "17_v18.py")
dio = _nap("dio", "19_drawio_chuong3.py")

from docx import Document  # noqa: E402

SUA_CHU = [
    ("Thiết kế kiến trúc cập nhật đồ thị cận thời gian thực",
     "hiện thực ứng dụng nguyên mẫu để truy vấn Neo4j, trực quan hóa đồ thị và kiểm tra các tệp kết quả thực nghiệm.",
     "hiện thực ứng dụng nguyên mẫu đọc trực tiếp bộ tệp xuất của đồ thị để truy vết hành vi, trực quan hóa đồ thị "
     "và kiểm tra các tệp kết quả thực nghiệm; Neo4j được dùng để nạp cùng bộ tệp này và kiểm tra lược đồ."),
    ("Ứng dụng nguyên mẫu cung cấp ba nhóm bằng chứng",
     "Ứng dụng nguyên mẫu cung cấp ba nhóm bằng chứng",
     "Ứng dụng nguyên mẫu và gói nạp Neo4j cung cấp ba nhóm bằng chứng"),
    ("Ứng dụng được hiện thực dưới dạng ứng dụng web cục bộ",
     "sử dụng Neo4j làm kho lưu trữ và truy vấn đồ thị tri thức.",
     "đọc trực tiếp bộ tệp xuất của đồ thị; Neo4j được dùng riêng để nạp cùng bộ tệp này và kiểm tra lược đồ đồ thị."),
]

# Thay trong tung run: co doan xen cong thuc, gop run se lam lech thu tu.
# Cum dai dat truoc cum ngan. "lop thong tin" (muc 3.3) la khai niem khac, khong dong vao.
TEN_DO_THI = [
    ("Node lưu vết tùy chọn trong Neo4j/POC", "Node tùy chọn của đồ thị truy vết trong Neo4j/POC"),
    ("Đồ thị lưu vết/POC", "Đồ thị truy vết/POC"),
    ("Đồ thị chiếu dùng để huấn luyện", "Đồ thị huấn luyện"),
    ("đồ thị chiếu dùng để huấn luyện", "đồ thị huấn luyện"),
    ("lớp lưu vết", "đồ thị truy vết"),
    ("graph lưu vết", "đồ thị truy vết"),
    ("lớp chiếu", "đồ thị huấn luyện"),
    ("graph huấn luyện", "đồ thị huấn luyện"),
    ("giữa hai lớp", "giữa hai đồ thị"),
]

CHANGELOG = [
    ("Mục 1.3.2, 3.7.2 và 4.5.2", "Sửa ba đoạn mô tả ứng dụng và Neo4j.",
     "Ứng dụng đọc trực tiếp bộ tệp CSV, không kết nối Neo4j; đoạn cũ nói ngược lại và mâu thuẫn với mục mô tả bước xuất."),
    ("Toàn quyển", "Thống nhất tên: đồ thị truy vết (giữ từng sự kiện) và đồ thị huấn luyện (gộp thành cạnh có trọng số).",
     "Bản cũ dùng lẫn lớp lưu vết, graph lưu vết, lớp chiếu, đồ thị chiếu, graph huấn luyện cho hai khái niệm."),
    ("Hình 3.1", "Vẽ lại luồng dữ liệu, có tệp draw.io.",
     "Hình cũ vẽ Neo4j cấp dữ liệu cho ứng dụng; thực tế cả hai cùng lấy từ bộ tệp CSV."),
]


def doi_ten_do_thi(doc) -> int:
    n = 0
    for para in v18.tat_ca_doan(doc):
        doi = False
        for run in para.runs:
            moi = run.text
            for cu, thay in TEN_DO_THI:
                n += moi.count(cu)
                moi = moi.replace(cu, thay)
            if moi != run.text:
                run.text = moi
                doi = True
        if doi:
            v18.TOUCHED.append(para)
    return n


def hinh_3_1() -> str:
    g = dio.Do()
    ink, muted, accent = dio.INK, dio.MUTED, dio.ACCENT

    def nhan(ten, *dong, color=muted):
        s = f'<b style="font-size:17px">{ten}</b>'
        for d in dong:
            s += f'<br><font style="font-size:14.5px" color="{color}">{d}</font>'
        return s

    box = f"rounded=1;arcSize=5;whiteSpace=wrap;html=1;strokeColor={ink};strokeWidth=1;fillColor=#FFFFFF;spacing=4;"
    lan = "rounded=1;arcSize=3;html=1;strokeColor=#C9CDD3;strokeWidth=1;fillColor=#F6F7F8;"
    den = f"rounded=1;strokeWidth=1.2;strokeColor={ink};"

    g.node(210, 0, 620, 240, "", lan, "lan_lv")
    g.node(210, 312, 620, 160, "", lan, "lan_ch")
    g.text(222, 6, 300, 22, "ĐỒ THỊ TRUY VẾT", size=12.5, align="left", color=muted, extra="fontStyle=1;spacingLeft=2;")
    g.text(222, 444, 420, 22, "ĐỒ THỊ HUẤN LUYỆN", size=12.5, align="left", color=muted,
           extra="fontStyle=1;spacingLeft=2;")

    data = g.node(0, 0, 172, 200, nhan("Dữ liệu<br>Retail Rocket", "events.csv", "item_properties", "category_tree"),
                  box + "fillColor=#F6F7F8;", "data")
    pre = g.node(0, 252, 172, 220, nhan("Tiền xử lý", "chia dữ liệu theo thời gian", "lọc thông tin sản phẩm",
                                        "đến mốc T<sub>train</sub>", "tạo ánh xạ định danh"), box, "pre")

    tep = g.node(222, 34, 164, 194, nhan("Bộ tệp đồ thị", "tệp CSV", "mỗi sự kiện", "là một dòng"), box, "tep")
    neo = g.node(654, 34, 164, 84, nhan("Neo4j", "kiểm tra lược đồ,", "số nút và cạnh"),
                 f"shape=cylinder3;boundedLbl=1;size=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor={ink};strokeWidth=1;",
                 "neo")
    app = g.node(654, 144, 164, 84, nhan("Ứng dụng", "xem lịch sử hành vi", "và danh sách gợi ý"), box, "app")

    a = g.node(222, 344, 164, 100, nhan("Trọng số cạnh", "W(u,i) = Σα<sub>b</sub>e<sup>−λΔt</sup>", color=accent),
               box + f"strokeColor={accent};strokeWidth=1.6;fillColor=#FCF1F0;", "a")
    gcn = g.node(438, 344, 164, 100, nhan("GCN 3 lớp", "lan truyền trên đồ thị", "học embedding bằng BPR"), box, "gcn")
    top = g.node(654, 344, 164, 100, nhan("Danh sách Top-K", "xếp hạng theo z<sub>u</sub>·z<sub>i</sub>"), box, "top")

    g.edge(data, pre, "", den + "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    g.edge(pre, tep, "", den + "exitX=1;exitY=0.1545;entryX=0;entryY=0.5;")
    g.edge(pre, a, "", den + "exitX=1;exitY=0.6455;entryX=0;entryY=0.5;")
    g.edge(tep, neo, "", den + "exitX=1;exitY=0.2165;entryX=0;entryY=0.5;")
    g.edge(tep, app, "", den + "exitX=1;exitY=0.7835;entryX=0;entryY=0.5;")
    g.text(420, 50, 200, 22, "nạp vào cơ sở dữ liệu", size=14)
    g.text(420, 160, 200, 22, "lịch sử hành vi", size=14)
    g.edge(a, gcn, "", den + "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    g.edge(gcn, top, "", den + "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    g.edge(top, app, "", den + "exitX=0.5;exitY=0;entryX=0.5;entryY=1;")
    g.text(744, 264, 120, 24, "kết quả đã lưu", size=14, align="left")
    g.edge(tep, a, "", f"rounded=1;strokeWidth=1.2;strokeColor={muted};dashed=1;dashPattern=4 3;endArrow=none;"
                       "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    g.text(312, 252, 340, 40, "<i>kiểm tra: tổng trọng số sự kiện<br>bằng trọng số cạnh</i>", size=14, align="left")
    g.text(0, 482, 830, 24, "<i>Màu đỏ: phần đóng góp của đề án (trọng số cạnh). "
                            "Kiến trúc lan truyền giữ nguyên LightGCN.</i>", size=14)
    return g.xml("Hinh 3.1")


def thay_hinh_3_1(doc) -> None:
    import io

    from docx.oxml.ns import qn
    from PIL import Image

    dio.render(hinh_3_1(), HINH_3_1, 880, 540)
    cap, blip = v18._blip_truoc_caption(doc, "Hình 3.1.")
    rid, _ = doc.part.get_or_add_image(io.BytesIO(HINH_3_1.read_bytes()))
    blip.set(qn("r:embed"), rid)
    w, h = Image.open(HINH_3_1).size
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
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "De_an_thac_si_v19.docx")
    ap.add_argument("--no-highlight", action="store_true")
    ap.add_argument("--bo-changelog", action="store_true", help="go trang danh muc sua doi -- ban gui co")
    args = ap.parse_args()

    v18.TOUCHED.clear()
    doc = Document(str(args.source))
    dem = (len(doc.element.body.findall(f".//{v18.OMML}oMath")), len(doc.tables), len(doc.inline_shapes))

    v18.xoa_to_nen(doc)
    for prefix, cu, moi in SUA_CHU:
        v18.bo_cum(doc, prefix, cu, moi)
    print(f"doi ten do thi: {doi_ten_do_thi(doc)} cho")
    thay_hinh_3_1(doc)

    bang = [t for t in doc.tables if t.rows and "Mục" in t.rows[0].cells[0].text]
    if len(bang) != 1:
        raise SystemExit(f"LOI: bang danh muc sua doi khop {len(bang)} lan, can 1")
    for muc, sua, ly_do in CHANGELOG:
        cells = bang[0].add_row().cells
        for cell, text in zip(cells, (f"[v19] {muc}", sua, ly_do)):
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
