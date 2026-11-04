"""v20 -> v21: demo da chay theo thiet ke, viet lai cac doan con ghi 'chua suy luan'.

Chay:
    python scripts/24_v21.py
    python scripts/24_v21.py --no-highlight --bo-changelog --out docs/De_an_thac_si_v21_ban_nop.docx

Ghi fig_v21_3_1_kien_truc.{drawio,png}.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "De_an_thac_si_v20.docx"
HINH_3_1 = ROOT / "docs" / "figures" / "fig_v21_3_1_kien_truc.png"

_spec = importlib.util.spec_from_file_location("v20", ROOT / "scripts" / "21_v20.py")
v20 = importlib.util.module_from_spec(_spec)
sys.modules["v20"] = v20
_spec.loader.exec_module(v20)
v18, dio = v20.v18, v20.dio

from docx import Document  # noqa: E402

GIAI_THICH = (
    "Các lần chạy của ma trận thực nghiệm không lưu trạng thái mô hình, nên danh sách Top-K ở trang truy vết là "
    "kết quả thật sinh ra trong quá trình đánh giá, được nạp lại từ tệp kết quả đã lưu (offline evaluation replay). "
    "Trang mô phỏng online dùng một lần huấn luyện riêng của BT-DKGRec-GCN (λ=0.05) trên nhóm Active, có lưu trạng "
    "thái mô hình: mỗi hành vi mới được ghi vào Neo4j và Top-K được suy luận lại theo lịch sử hiện tại của khách "
    "mà không huấn luyện lại. Lần chạy này chỉ minh họa luồng hệ thống, không thuộc các bảng kết quả."
)

SUA_CHU = [
    ("Thiết kế kiến trúc cập nhật đồ thị cận thời gian thực",
     "hiện thực ứng dụng nguyên mẫu đọc trực tiếp bộ tệp xuất của đồ thị để truy vết hành vi, trực quan hóa đồ thị "
     "và kiểm tra các tệp kết quả thực nghiệm; Neo4j được dùng để nạp cùng bộ tệp này và kiểm tra lược đồ.",
     "hiện thực ứng dụng nguyên mẫu gồm trang truy vết đọc bộ tệp xuất của đồ thị và các tệp kết quả thực nghiệm, "
     "cùng trang mô phỏng online ghi hành vi mới vào Neo4j rồi tính lại Top-K."),
    ("Bước xuất dữ liệu sinh ra một bộ tệp CSV",
     "vừa là nguồn đọc trực tiếp của ứng dụng nguyên mẫu, nên phần trình diễn không phụ thuộc vào một dịch vụ cơ sở dữ liệu đang chạy.",
     "vừa là nguồn đọc trực tiếp của trang truy vết, nên trang này không phụ thuộc vào một dịch vụ cơ sở dữ liệu đang chạy; "
     "trang mô phỏng online thì cần Neo4j đang chạy."),
    ("Ứng dụng được hiện thực dưới dạng ứng dụng web cục bộ",
     "đọc trực tiếp bộ tệp xuất của đồ thị; Neo4j được dùng riêng để nạp cùng bộ tệp này và kiểm tra lược đồ đồ thị.",
     "gồm hai trang: trang truy vết đọc trực tiếp bộ tệp xuất của đồ thị, trang mô phỏng online đọc và ghi đồ thị trong Neo4j."),
    ("Danh sách Top-K trên giao diện là kết quả thật của mô hình BT-DKGRec-GCN, được nạp lại",
     "Danh sách Top-K trên giao diện là kết quả thật của mô hình BT-DKGRec-GCN, được nạp lại từ tệp kết quả đánh giá đã lưu "
     "(offline evaluation replay). Ứng dụng chưa nạp trạng thái mô hình để suy luận theo từng yêu cầu vì trọng số không được "
     "lưu checkpoint trong quá trình huấn luyện.",
     "Trang truy vết nạp lại Top-K từ tệp kết quả đánh giá đã lưu (offline evaluation replay). Trang mô phỏng online suy luận "
     "theo từng yêu cầu nhưng mới dùng một lần huấn luyện riêng trên nhóm Active, Neo4j chỉ chứa 234 khách của nhóm này, "
     "và việc huấn luyện lại theo chu kỳ chưa được tự động hóa."),
    ("Kết quả ủng hộ có điều kiện vai trò của behavior-time",
     "suy luận trực tuyến và đánh giá trong môi trường vận hành là hướng phát triển tiếp theo.",
     "đánh giá trong môi trường vận hành là hướng phát triển tiếp theo."),
    ("Ứng dụng chưa nạp trạng thái BT-DKGRec-GCN để suy luận trực tiếp",
     "Ứng dụng chưa nạp trạng thái BT-DKGRec-GCN để suy luận trực tiếp theo từng yêu cầu. Danh sách Top-K trên giao diện là "
     "kết quả thật của mô hình, được nạp lại từ tệp kết quả đã lưu; việc nạp trạng thái mô hình để suy luận real-time là bước "
     "hoàn thiện tiếp theo.",
     "Trang truy vết hiển thị Top-K thật của mô hình, được nạp lại từ tệp kết quả đã lưu. Trang mô phỏng online nạp trạng thái "
     "của một lần huấn luyện riêng trên nhóm Active: hành vi mới được ghi vào Neo4j và Top-K được tính lại ngay; việc mở rộng "
     "sang nhóm đánh giá chính là bước hoàn thiện tiếp theo."),
    ("Kiến trúc ứng dụng phân tách lớp tiếp nhận",
     "nhưng chưa bao gồm dịch vụ suy luận, giám sát vận hành hoặc kiểm thử khả năng mở rộng.",
     "nhưng dịch vụ suy luận mới ở mức mô phỏng và chưa bao gồm giám sát vận hành hoặc kiểm thử khả năng mở rộng."),
    ("Ứng dụng nguyên mẫu chưa lưu và nạp trạng thái mô hình",
     "Ứng dụng nguyên mẫu chưa lưu và nạp trạng thái mô hình BT-DKGRec-GCN để suy luận theo từng yêu cầu.",
     "Trang mô phỏng online mới chạy trên nhóm Active với một lần huấn luyện riêng, chưa tự động huấn luyện lại theo chu kỳ."),
    ("Hướng phát triển tự nhiên của ứng dụng là lưu trạng thái mô hình",
     "lưu trạng thái mô hình cùng ánh xạ người dùng-sản phẩm để danh sách Top-K trên giao diện phản ánh trực tiếp kết quả suy luận của BT-DKGRec-GCN.",
     "mở rộng trang mô phỏng online sang nhóm đánh giá chính và tự động hóa việc huấn luyện lại từ ảnh chụp đồ thị."),
    ("Giá trị của đề án nằm ở sự kết hợp",
     "dịch vụ suy luận mô hình và hiệu quả vận hành là hướng phát triển tiếp theo.",
     "hiệu quả vận hành của dịch vụ suy luận là hướng phát triển tiếp theo."),
]

CHANGELOG = [
    ("Mục 1.3.2, 3.7.2, 4.5.2, 4.5.6, 4.6, 5.1.3, 5.2.2, 5.3.3, 5.4.3, 5.5, Bảng 4.12",
     "Mô tả hai trang: trang truy vết phát lại kết quả, trang mô phỏng online suy luận theo hành vi mới.",
     "Demo đã chạy theo thiết kế mục 3.7.1 với một lần huấn luyện riêng có lưu mô hình."),
    ("Hình 3.1", "Chỉ còn nhánh ảnh chụp đồ thị là nét đứt.", "Các nhánh còn lại đã hiện thực trong trang mô phỏng online."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "De_an_thac_si_v21.docx")
    ap.add_argument("--no-highlight", action="store_true")
    ap.add_argument("--bo-changelog", action="store_true", help="go trang danh muc sua doi -- ban gui co")
    args = ap.parse_args()

    v18.TOUCHED.clear()
    doc = Document(str(args.source))
    dem = (len(doc.element.body.findall(f".//{v18.OMML}oMath")), len(doc.tables), len(doc.inline_shapes))
    v18.xoa_to_nen(doc)

    for prefix, cu, moi in SUA_CHU:
        v18.bo_cum(doc, prefix, cu, moi)
    doan = v18.para_bat_dau(doc, "Ứng dụng nguyên mẫu và gói nạp Neo4j cung cấp ba nhóm bằng chứng")
    m = re.search(r"Tuy nhiên, kho mã nguồn chưa lưu.*?từ trạng thái mô hình\.", doan.text)
    if m is None:
        raise SystemExit("LOI: khong thay doan gioi han suy luan o muc 3.7.2")
    v18.dat_text(doan, doan.text.replace(m.group(0), GIAI_THICH))

    dong = [r for t in doc.tables for r in t.rows if r.cells[0].text.strip() == "Dịch vụ suy luận mô hình"]
    if len(dong) != 1:
        raise SystemExit(f"LOI: dong Dich vu suy luan khop {len(dong)} lan, can 1")
    for cell, text in zip(dong[0].cells[1:], (
            "Trang mô phỏng online nạp trạng thái của một lần huấn luyện riêng trên nhóm Active",
            "Luồng ghi hành vi và tính lại Top-K chạy được; chưa phải hệ thống vận hành, số chính thức vẫn ở các bảng kết quả")):
        v18.dat_text(cell.paragraphs[0], text)

    dio.render(v20.hinh_3_1(da_noi=True), HINH_3_1, 930, 560)
    v20.thay_anh(doc, "Hình 3.1.", HINH_3_1)

    bang = [t for t in doc.tables if t.rows and "Mục" in t.rows[0].cells[0].text]
    if len(bang) != 1:
        raise SystemExit(f"LOI: bang danh muc sua doi khop {len(bang)} lan, can 1")
    for muc, sua, ly_do in CHANGELOG:
        cells = bang[0].add_row().cells
        for cell, text in zip(cells, (f"[v21] {muc}", sua, ly_do)):
            cell.text = text
            v18.TOUCHED.extend(cell.paragraphs)

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
