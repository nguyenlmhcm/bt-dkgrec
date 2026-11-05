"""v21 -> v22: quyen mo ta dung demo ba phan he, sua cac cau lech voi bang so, Bang 4.9 va Hinh 4.5 dung lambda=0.05.

Chay:
    python scripts/27_v22.py
    python scripts/27_v22.py --no-highlight --bo-changelog --out docs/De_an_thac_si_v22_ban_nop.docx

Ba anh giao dien fig_v22_4_{7,8,9}_*.png chup tu demo that (xem D39). So o Bang 4.9 doc tu experiments/runs/.
So trang trong muc luc lay tu ban render LibreOffice; mo bang Word van phai F9 de chot.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import io
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "docs" / "De_an_thac_si_v21.docx"
FIG = ROOT / "docs" / "figures"


def _nap(ten: str, file: str):
    spec = importlib.util.spec_from_file_location(ten, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[ten] = mod
    spec.loader.exec_module(mod)
    return mod


v21 = _nap("v21", "24_v21.py")
v20, v18 = v21.v20, v21.v18
bd = _nap("bieu_do", "22_bieu_do_chuong4.py")

from docx import Document  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.table import Table  # noqa: E402
from docx.text.paragraph import Paragraph  # noqa: E402

from src.evaluation.reporting import load_runs  # noqa: E402

UNG_DUNG_VI = (
    "Sản phẩm ứng dụng của đề tài là một ứng dụng web mô phỏng giải pháp gợi ý cho một cửa hàng trên sàn thương mại "
    "điện tử, gồm ba phân hệ đăng nhập riêng. Khách hàng xem sản phẩm, thêm vào giỏ, mua hàng và nhận danh sách gợi ý "
    "được tính lại theo hành vi vừa phát sinh. Cửa hàng quản lý danh mục và sản phẩm, theo dõi hành vi của khách theo "
    "thời gian và nhận gợi ý khách nên chăm sóc. Nhà cung cấp giải pháp quan sát đồ thị tri thức động, đường lan truyền "
    "và hiệu quả mô hình, đồng thời quản lý các phiên bản mô hình; bước huấn luyện lại theo chu kỳ hiện chưa được tự "
    "động hóa."
)
UNG_DUNG_EN = (
    "The applied outcome of the project is a web application that simulates the recommendation solution for a store on "
    "an e-commerce platform, with three separately authenticated portals. Customers browse, add items to the cart, place "
    "orders, and receive recommendations recomputed from the behavior they have just produced. The store manages "
    "categories and products, follows each customer's behavior over time, and receives suggestions on which customers to "
    "approach. The solution provider inspects the dynamic knowledge graph, the propagation paths, and model performance, "
    "and manages model versions; periodic retraining is not yet automated."
)

# (dau doan, noi dung moi) -- thay ca doan.
THAY_DOAN = [
    ("Sản phẩm ứng dụng của đề tài là một chương trình web", UNG_DUNG_VI),
    ("The applied outcome of the project is a web program", UNG_DUNG_EN),
    ("Ứng dụng nguyên mẫu được xây dựng để truy vết lịch sử hành vi",
     "Ứng dụng nguyên mẫu được xây dựng để mô phỏng luồng gợi ý cho một cửa hàng ở ba phân hệ khách hàng, cửa hàng và "
     "nhà cung cấp giải pháp, truy vết lịch sử hành vi, trực quan hóa đồ thị tri thức động và trình bày kết quả thực nghiệm."),
    ("Ứng dụng cho phép chọn người dùng, xem lịch sử tương tác",
     "Ứng dụng nguyên mẫu mô phỏng tình huống đơn vị cung cấp giải pháp triển khai hệ thống gợi ý cho một cửa hàng trên sàn "
     "thương mại điện tử, gồm ba phân hệ có đăng nhập và phân quyền riêng: phân hệ khách hàng, phân hệ cửa hàng và phân hệ "
     "nhà cung cấp giải pháp. Danh mục, sản phẩm và hành vi lấy từ dữ liệu Retail Rocket; do dữ liệu gốc đã ẩn danh, tên "
     "sản phẩm, tên khách, nhãn hàng và một phần giá được sinh để minh họa và giao diện có ghi rõ điều này. Ngoài ba phân hệ, "
     "trang truy vết được giữ làm công cụ nội bộ để đối chiếu đồ thị truy vết, đồ thị huấn luyện và danh sách Top-K với các "
     "tệp kết quả thực nghiệm."),
    ("Phạm vi hiện thực trên chưa cấu thành một hệ thống vận hành hoàn chỉnh",
     "Phạm vi hiện thực trên chưa cấu thành một hệ thống vận hành hoàn chỉnh. Đề án chưa đánh giá độ trễ, thông lượng, A/B "
     "testing hoặc cơ chế cập nhật mô hình trong môi trường thực tế. Ứng dụng đã hiện thực lớp tiếp nhận sự kiện, dịch vụ "
     "cập nhật đồ thị, kho đồ thị và dịch vụ dự báo Top-K; lớp huấn luyện ngoại tuyến mới chạy thủ công, chưa có ảnh chụp đồ "
     "thị gộp hành vi, khách hàng và sản phẩm mới, cũng chưa có lịch huấn luyện tự động."),
    ("Cấu hình này khác bản v11 ở ba điểm",
     "Ba lựa chọn về ngân sách huấn luyện, số seed và kích thước mô hình được giải thích như sau."),
    ("Thứ nhất, mô hình được huấn luyện đến khi validation không còn cải thiện",
     "Thứ nhất, mô hình được huấn luyện với cơ chế dừng sớm theo validation và trần 1000 epoch. Trên original split, các mô "
     "hình đạt validation NDCG@20 tốt nhất trong khoảng epoch 50 đến 145, riêng một lần chạy của mô hình λ=0.05 ở epoch 400. "
     "Trên active split, 10 trong 12 lần chạy chạm trần 1000 epoch với epoch tốt nhất từ 760 đến 985. Một so sánh trong đó "
     "mọi mô hình đều chưa hội tụ phản ánh tốc độ hội tụ trong giai đoạn đầu chứ không phản ánh chất lượng cuối cùng."),
    ("Bỏ tiếp tri thức sản phẩm thì thu được LightGCN",
     "Bỏ tiếp tri thức sản phẩm thì thu được LightGCN, vốn chỉ dùng đồ thị tương tác giữa người dùng và sản phẩm. Recall@20 "
     "giảm tiếp xuống 0.023736 trên nhóm thứ nhất và 0.028601 trên nhóm thứ hai. Trên nhóm thứ nhất, hai thành phần đóng góp "
     "xấp xỉ nhau (0.003442 của tri thức sản phẩm và 0.003275 của trọng số hành vi-thời gian); trên nhóm thứ hai, trọng số "
     "hành vi-thời gian đóng góp phần lớn hơn (0.006687 so với 0.002716). Đóng góp của tri thức sản phẩm lặp lại ở cả ba seed "
     "trên cả hai nhóm, trong khi đóng góp của trọng số hành vi-thời gian lặp lại ở hai trong ba seed."),
    ("Với ba lần chạy cho mỗi cấu hình, kết luận của chương dựa trên",
     "Với ba lần chạy cho mỗi cấu hình, kết luận của chương dựa trên vị trí đứng đầu của mô hình đề xuất: vị trí này giữ nguyên "
     "trên cả hai nhóm người dùng và cả bốn chỉ số đánh giá, còn thứ tự giữa các mô hình xếp dưới có thay đổi theo nhóm và theo chỉ số."),
    ("Khoảng cách giữa hai mô hình trong nhóm so sánh này",
     "Giá trị trung bình của mô hình đề xuất cao hơn Static KG-GCN ở cả ba chỉ số trên cả hai nhóm người dùng, nhưng chênh lệch "
     "không lặp lại ở mọi seed: chỉ HitRate@20 trên nhóm người dùng tích cực cao hơn ở cả ba lần chạy, còn NDCG@20 trên nhóm có "
     "lịch sử ban đầu chỉ cao hơn ở một lần chạy. Trên bộ dữ liệu này, đề án chưa chứng minh được rằng riêng trọng số hành "
     "vi-thời gian cải thiện ổn định độ chính xác xếp hạng so với đồ thị tri thức tĩnh."),
    ("Chênh lệch mang dấu dương ở phần lớn các phép so sánh",
     "Với λ=0.01, giá trị kế thừa chưa qua bước dò, chênh lệch Recall@20 so với Static KG-GCN chỉ còn 5,89% trên nhóm có lịch sử "
     "ban đầu và 3,60% trên nhóm tích cực (mục 4.2.4). Kết quả vì vậy nhất quán về chiều ở giá trị trung bình nhưng biên độ phần "
     "lớn nằm trong dao động giữa các seed, và kết luận phù hợp là chưa đủ bằng chứng để khẳng định hiệu ứng, không phải bằng "
     "chứng cho thấy không có hiệu ứng."),
    ("Ứng dụng nguyên mẫu được xây dựng để minh họa luồng dữ liệu",
     "Ứng dụng nguyên mẫu được xây dựng để minh họa luồng dữ liệu của kiến trúc ở mục 3.7.1 trong bối cảnh sử dụng và kiểm chứng "
     "các kết quả nghiên cứu. Mục tiêu không phải xây dựng một website thương mại điện tử hoàn chỉnh, mà là cho thấy mỗi nhóm "
     "người dùng khai thác kết quả dự báo như thế nào, đồng thời cung cấp bằng chứng trực quan về lược đồ đồ thị tri thức động, "
     "trọng số cạnh, đường lan truyền và kết quả đánh giá."),
    ("Ứng dụng được hiện thực dưới dạng ứng dụng web cục bộ",
     "Ứng dụng được hiện thực dưới dạng ứng dụng web gồm dịch vụ nền FastAPI và giao diện HTML, CSS, JavaScript; Neo4j là kho đồ "
     "thị tri thức động; SQLite lưu tài khoản, giỏ hàng, đơn hàng mô phỏng, chiến dịch ưu đãi và phần chỉnh sửa danh mục. Mô hình "
     "suy luận trên CPU, không cần GPU khi phục vụ. Trang truy vết là một ứng dụng riêng, đọc trực tiếp bộ tệp xuất của đồ thị."),
    ("Các chức năng chính gồm:", "Các chức năng chính được tổ chức theo ba phân hệ:"),
    ("chọn hoặc nhập visitor để xem lịch sử;",
     "Phân hệ khách hàng: xem và tìm sản phẩm, thêm vào giỏ, đặt đơn mô phỏng, xem lịch sử hành vi và giao dịch; nhận danh sách "
     "gợi ý cá nhân kèm lý do, cùng các mục khách khác cũng quan tâm và thường mua kèm."),
    ("hiển thị đồ thị con Visitor-Item-Category/Property;",
     "Phân hệ cửa hàng: thêm, sửa, xóa danh mục và sản phẩm; xem khách ghé thăm và hành vi của từng khách theo thời gian ghi nhận; "
     "xem khách bỏ giỏ, khách nên chăm sóc theo từng sản phẩm, gửi ưu đãi và theo dõi sản phẩm được quan tâm."),
    ("xem số sự kiện, trọng số cạnh và bằng chứng hành vi;",
     "Phân hệ nhà cung cấp giải pháp: xem đồ thị theo từng khách, lược đồ và số lượng của toàn đồ thị, đường lan truyền từ khách "
     "đến sản phẩm gợi ý; đưa sự kiện thật tiếp theo vào đồ thị để quan sát trọng số và Top-K thay đổi; xem hiệu quả mô hình, phát "
     "lại dữ liệu kiểm thử, API tích hợp; áp dụng hoặc quay lại phiên bản mô hình."),
    ("xem các chỉ số đánh giá của các mô hình;",
     "Phân quyền: bản trình diễn có hai tài khoản khách hàng, một tài khoản cửa hàng và một tài khoản nhà cung cấp giải pháp; mỗi "
     "vai có phiên đăng nhập riêng và không đọc được dữ liệu của vai khác."),
    ("duy trì khả năng trình diễn bằng bộ dữ liệu mẫu",
     "Dữ liệu quá khứ: 234 khách của nhóm Active với 24.892 sự kiện và 1.051 lượt mua thật của tập huấn luyện; hành vi phát sinh "
     "khi trình diễn được đánh dấu riêng và có thể xóa để đưa dữ liệu về ban đầu."),
    ("Giao diện gồm khu vực chọn người dùng, bảng lịch sử",
     "Hình 4.7, Hình 4.8 và Hình 4.9 trình bày giao diện của ba phân hệ. Lý do gợi ý được đọc từ đường đi ba bước trong ma trận "
     "kề chuẩn hóa Â (khách hàng, sản phẩm đã tương tác, nút trung gian, sản phẩm gợi ý), trong đó nút trung gian có thể là khách "
     "hàng khác, danh mục hoặc thuộc tính. Đường đi là một thành phần của Â³, phản ánh ảnh hưởng lan truyền chứ không phải phân rã "
     "của điểm dự báo, vì điểm còn phụ thuộc vào biểu diễn đã học."),
    ("Yêu cầu từ giao diện được chuyển thành truy vấn trên kho đồ thị",
     "Khi khách hàng phát sinh hành vi, giao diện gọi lớp tiếp nhận sự kiện; hành vi được ghi vào Neo4j kèm thời gian ghi nhận và "
     "trọng số. Khi cần danh sách gợi ý, dịch vụ dự báo đọc lịch sử hiện tại của khách từ Neo4j, tính trọng số cạnh theo các công "
     "thức (3.6)-(3.8) tại mốc thời gian τ hiện tại, tính lại biểu diễn phía khách hàng (fold-in) và xếp hạng trên biểu diễn phía "
     "sản phẩm giữ từ lần huấn luyện, sau khi loại các sản phẩm khách đã tương tác. Do mô hình đã học đến mốc kết thúc tập huấn "
     "luyện, ứng dụng chỉ đi tới theo thời gian chứ không quay về quá khứ: hành vi mới trên giao diện được đặt ngay sau mốc này; "
     "chức năng sự kiện thật tiếp theo đưa các sự kiện của giai đoạn xác thực và kiểm thử qua cùng đường ghi theo đúng thời điểm "
     "gốc, khi đó τ tiến lên và trọng số của các hành vi cũ giảm theo hàm suy giảm mũ."),
    ("Ứng dụng nguyên mẫu đáp ứng mục tiêu truy vết",
     "Ứng dụng nguyên mẫu đáp ứng mục tiêu minh họa luồng từ hành vi đến gợi ý ở ba phân hệ, truy vết, trực quan hóa và kiểm chứng "
     "các tệp kết quả. Khi chưa có hành vi mới, Top-K do dịch vụ dự báo tính khớp tệp kết quả của chính lần huấn luyện này trên "
     "234/234 khách. Phép phát lại 1.000 sự kiện đầu của giai đoạn kiểm thử, gồm 49 khách, theo đúng thứ tự thời gian qua đường "
     "ghi đồ thị cho thấy ở lần đầu khách chạm một sản phẩm, sản phẩm đó đã có sẵn trong Top-20 của mô hình ở 26/492 lần (5,3%), "
     "so với 6/492 lần (1,2%) của danh sách bán chạy; phép đo này chỉ minh họa trên bản huấn luyện CPU, khác cách đo ở mục 4.2 và "
     "không thay thế các bảng kết quả."),
    ("Trang truy vết nạp lại Top-K từ tệp kết quả đánh giá đã lưu",
     "Ba phân hệ mới dùng một lần huấn luyện riêng trên CPU với nhóm Active, đạt Recall@20 0.0317 trên tập kiểm thử, thấp hơn "
     "0.044177 của lần chạy cùng seed ở Bảng 4.7, và kho Neo4j chỉ chứa 234 khách của nhóm này. Mức thay đổi của Top-K phụ thuộc "
     "vào độ dày lịch sử: khảo sát một lần thêm giỏ trên khoảng 170 sản phẩm cho thấy với khách có 36 sự kiện, Top-20 chủ yếu đổi "
     "thứ hạng và có thêm tối đa 2 sản phẩm mới; với khách có 7 sự kiện, trung vị là 4 sản phẩm mới và tối đa là 15. Danh mục và "
     "sản phẩm do cửa hàng chỉnh sửa nằm ở một lớp riêng, chưa đi vào đồ thị huấn luyện, nên sản phẩm mới bán được nhưng chưa được "
     "gợi ý. Xếp sản phẩm theo số khách có sản phẩm đó trong Top-20 chưa tốt hơn danh sách bán chạy trong việc đoán nhu cầu thật "
     "(100 sản phẩm đầu chiếm 4,5% so với 7,1%), nên ứng dụng chỉ trình bày là sản phẩm được quan tâm, không dùng để dự báo nhập "
     "hàng. Việc huấn luyện lại theo chu kỳ chưa được tự động hóa và ứng dụng chưa có thanh toán thật. Vì vậy, ứng dụng được xác "
     "định là nguyên mẫu nghiên cứu phục vụ kiểm chứng và trình bày kết quả; việc hoàn thiện thành hệ thống vận hành nằm ngoài phạm "
     "vi đề án và được thảo luận ở mục 5.4."),
    ("Chương 4 đã trình bày dữ liệu, giao thức, cấu hình",
     "Chương 4 đã trình bày dữ liệu, giao thức, cấu hình, kết quả hai nhóm, ablation, so sánh mô hình, phân tích hạn chế và ứng dụng "
     "nguyên mẫu. Trên cả hai nhóm người dùng, BT-DKGRec-GCN với λ=0.05 đạt giá trị trung bình cao nhất ở cả bốn chỉ số tại K = 20 và "
     "vượt LightGCN ở Recall@20 và HitRate@20 trong cả ba lần chạy; khoảng cách so với Static KG-GCN chưa ổn định giữa các seed."),
    ("Trang truy vết hiển thị Top-K thật của mô hình",
     "Ba phân hệ nạp trạng thái của một lần huấn luyện riêng trên nhóm Active: hành vi mới được ghi vào Neo4j và Top-K được tính lại "
     "ngay. Trang truy vết hiển thị Top-K thật của các lần chạy, được nạp lại từ tệp kết quả đã lưu. Việc mở rộng kho đồ thị sang nhóm "
     "đánh giá chính và tự động hóa huấn luyện lại là bước hoàn thiện tiếp theo."),
    ("minh họa kiến trúc cập nhật đồ thị cận thời gian thực",
     "mô phỏng cách khách hàng, cửa hàng và nhà cung cấp giải pháp khai thác kết quả dự báo trên kiến trúc cập nhật đồ thị cận thời gian thực."),
    ("Trang mô phỏng online mới chạy trên nhóm Active",
     "Các phân hệ mới chạy trên nhóm Active với một lần huấn luyện riêng, chưa tự động huấn luyện lại theo chu kỳ; danh mục và sản "
     "phẩm mới chưa đi vào đồ thị huấn luyện. Hiệu năng phục vụ và tác động kinh doanh trực tuyến chưa được đo lường trong phạm vi đề án."),
    ("Coverage thấp: BT-DKGRec-GCN có xu hướng",
     "Coverage thấp: danh sách gợi ý của các mô hình đồ thị chỉ phủ khoảng 2% đến 3,5% tập item ứng viên tại K = 20."),
    ("Kết quả phụ thuộc nhóm: lợi thế rõ nhất",
     "Kết quả phụ thuộc nhóm: đóng góp của suy giảm thời gian rõ hơn trên nhóm người dùng tích cực."),
    ("Coverage@20 của mô hình đề xuất thấp hơn một số mô hình đối chứng",
     "Coverage@20 trung bình của mô hình đề xuất cao nhất trên cả hai nhóm nhưng thấp hơn Static KG-GCN ở một số lần chạy, và giá trị "
     "tuyệt đối của mọi mô hình đều thấp. Đề án báo cáo chỉ số này đúng như đo được và không xem độ đa dạng của danh sách gợi ý là mục "
     "tiêu tối ưu."),
    ("Trên nhóm người dùng có lịch sử ban đầu, BT-DKGRec-GCN đạt kết quả cao hơn các đối chứng đơn giản",
     "Trên cả hai nhóm người dùng, BT-DKGRec-GCN với λ=0.05 đạt giá trị trung bình cao nhất ở cả bốn chỉ số tại K = 20 và vượt LightGCN "
     "ở Recall@20 trong cả ba lần chạy. Bằng chứng này ủng hộ có điều kiện vai trò của mô hình hóa đồ thị hành vi-thời gian, nhưng chưa "
     "đủ để kết luận về ưu thế tổng quát."),
]

# (dau doan, cum cu, cum moi)
SUA_CUM = [
    ("NGÀY HOÀN THÀNH NHIỆM VỤ", "năm 2025", "năm 2026"),
    ("Tp. Hồ Chí Minh, ngày", "năm 2025", "năm 2026"),
    ("Tôi xin khẳng định rằng đề án", "học thuật.Tôi", "học thuật. Tôi"),
    ("Thiết kế kiến trúc cập nhật đồ thị cận thời gian thực",
     "hiện thực ứng dụng nguyên mẫu gồm trang truy vết đọc bộ tệp xuất của đồ thị và các tệp kết quả thực nghiệm, cùng trang mô phỏng "
     "online ghi hành vi mới vào Neo4j rồi tính lại Top-K.",
     "hiện thực ứng dụng nguyên mẫu mô phỏng giải pháp gợi ý cho một cửa hàng với ba phân hệ khách hàng, cửa hàng và nhà cung cấp giải "
     "pháp, trong đó hành vi mới được ghi vào Neo4j rồi Top-K được tính lại; trang truy vết đối chiếu đồ thị với các tệp kết quả thực nghiệm."),
    ("Bước xuất dữ liệu sinh ra một bộ tệp CSV", "trang mô phỏng online thì cần Neo4j đang chạy", "ba phân hệ của ứng dụng thì cần Neo4j đang chạy"),
    ("Ứng dụng nguyên mẫu và gói nạp Neo4j cung cấp ba nhóm bằng chứng",
     "Trang mô phỏng online dùng một lần huấn luyện riêng của BT-DKGRec-GCN (λ=0.05) trên nhóm Active, có lưu trạng thái mô hình: mỗi "
     "hành vi mới được ghi vào Neo4j và Top-K được suy luận lại theo lịch sử hiện tại của khách mà không huấn luyện lại.",
     "Ba phân hệ dùng một lần huấn luyện riêng của BT-DKGRec-GCN (λ=0.05) trên nhóm Active, có lưu trạng thái mô hình: mỗi hành vi mới "
     "đi qua lớp tiếp nhận, được ghi vào Neo4j, rồi biểu diễn phía khách hàng được tính lại từ lịch sử hiện tại trong khi biểu diễn phía "
     "sản phẩm giữ từ lần huấn luyện, nên Top-K được suy luận lại mà không huấn luyện lại."),
    ("Thứ hai, mỗi cấu hình được chạy ba seed", "ba seed thay vì một", "ba seed"),
    ("Hiệu ứng nhỏ của behavior-time weighting", "Hiệu ứng nhỏ của behavior-time weighting có thể",
     "Việc chênh lệch do behavior-time weighting chưa ổn định giữa các seed có thể"),
    ("Thứ tự giữa các mô hình giữ nguyên", "Thứ tự giữa các mô hình giữ nguyên trên cả hai nhóm người dùng và cả bốn chỉ số đánh giá, và mô hình đề xuất vượt LightGCN ở cả ba lần chạy.",
     "Mô hình đề xuất đứng đầu trên cả hai nhóm người dùng và cả bốn chỉ số đánh giá, và vượt LightGCN ở Recall@20 trong cả ba lần chạy; thứ tự giữa các mô hình xếp dưới có thay đổi theo nhóm và theo chỉ số."),
    ("Ý nghĩa của kết quả nằm ở cách bố trí thí nghiệm", "Thứ tự giữa các mô hình giữ nguyên trên cả hai nhóm người dùng và cả bốn chỉ số đánh giá, và mô hình đề xuất vượt LightGCN ở cả ba lần chạy.",
     "Mô hình đề xuất đứng đầu trên cả hai nhóm người dùng và cả bốn chỉ số đánh giá, và vượt LightGCN ở Recall@20 trong cả ba lần chạy."),
    ("Ba kết quả chính có thể rút ra", "vượt xa các mô hình đối chứng dựa trên độ phổ biến, với mức cải thiện khoảng 98% trên nhóm người dùng có lịch sử ban đầu",
     "vượt xa các mô hình đối chứng dựa trên độ phổ biến; trên nhóm người dùng có lịch sử ban đầu, riêng LightGCN đã cao hơn Popularity khoảng 98%"),
    ("Mỗi cấu hình được chạy ba seed với cơ chế dừng sớm", "các mô hình đối chứng dùng cấu hình theo công bố gốc", "các mô hình đối chứng dùng cấu hình cố định"),
    ("Mỗi cấu hình được chạy ba lần với ba seed khác nhau",
     "nên chênh lệch của từng cặp mô hình liền kề chưa đạt ngưỡng ý nghĩa thống kê thông thường", "nên chênh lệch của từng cặp mô hình liền kề chưa đủ để kết luận riêng"),
    ("Mỗi cấu hình được chạy ba lần với ba seed khác nhau",
     "dựa trên tính nhất quán của thứ tự giữa các mô hình trên cả hai nhóm người dùng và cả bốn chỉ số",
     "dựa trên vị trí đứng đầu của mô hình đề xuất trên cả hai nhóm người dùng và cả bốn chỉ số"),
    ("Kiến trúc ứng dụng phân tách lớp tiếp nhận", "dịch vụ suy luận mới ở mức mô phỏng", "dịch vụ suy luận mới ở mức nguyên mẫu"),
    ("Kết quả ủng hộ có điều kiện vai trò của behavior-time", "Ứng dụng nguyên mẫu bổ sung lớp kiểm chứng dữ liệu và đồ thị",
     "Ứng dụng nguyên mẫu minh họa luồng sử dụng ở ba phân hệ và bổ sung lớp kiểm chứng dữ liệu và đồ thị"),
    ("Hướng phát triển tự nhiên của ứng dụng", "mở rộng trang mô phỏng online sang nhóm đánh giá chính và tự động hóa việc huấn luyện lại từ ảnh chụp đồ thị.",
     "hoàn thiện bước huấn luyện lại theo chu kỳ: tạo ảnh chụp đồ thị gộp hành vi, khách hàng và sản phẩm mới, huấn luyện ngoại tuyến, so sánh "
     "trên tập xác thực rồi áp dụng phiên bản đạt yêu cầu; đồng thời mở rộng kho đồ thị sang nhóm đánh giá chính."),
    ("Có thể phát triển cơ chế giải thích dựa trên đường đi", "Có thể phát triển cơ chế giải thích dựa trên đường đi Visitor-Item-Category/Property,",
     "Ứng dụng hiện đọc đường lan truyền Visitor-Item-Category/Property ở mức minh họa; có thể phát triển thành cơ chế giải thích có phân rã điểm dự báo,"),
    ("Giá trị của đề án nằm ở sự kết hợp", "ứng dụng nguyên mẫu hỗ trợ truy vết", "ứng dụng nguyên mẫu ba phân hệ hỗ trợ khai thác và truy vết"),
]

XOA_DOAN = ("Kết quả này khác kết luận trong bản v11", "Kết quả này cũng khác kết luận trong bản v11")

BANG_4_12 = [
    ("Phân hệ khách hàng", "Hai tài khoản xem, thêm giỏ, mua; hành vi vào đồ thị truy vết; Top-K tính lại theo lịch sử hiện tại",
     "Luồng từ hành vi đến gợi ý chạy được trên 234 khách nhóm Active; chưa phải hệ thống vận hành"),
    ("Phân hệ cửa hàng", "Thêm, sửa, xóa danh mục và sản phẩm; hành vi của khách theo thời gian; khách nên chăm sóc; sản phẩm được quan tâm",
     "Cửa hàng khai thác được kết quả dự báo; sản phẩm mới chưa được gợi ý khi chưa huấn luyện lại; không dùng để dự báo nhập hàng"),
    ("Phân hệ nhà cung cấp giải pháp", "Đồ thị theo khách, toàn đồ thị, đường lan truyền, phiên bản mô hình, API tích hợp",
     "Quan sát được đồ thị thay đổi theo sự kiện; đường lan truyền không phải phân rã điểm dự báo"),
    ("Phát lại dữ liệu thật", "1.000 sự kiện đầu giai đoạn kiểm thử: 26/492 lần chạm đầu có sẵn trong Top-20, bán chạy 6/492",
     "Minh họa trên bản huấn luyện CPU, khác cách đo ở mục 4.2"),
    ("Truy vết và kết quả thực nghiệm", "Gói nạp Neo4j kèm số liệu đối chiếu; trang truy vết; tệp chỉ số và danh sách gợi ý của các lần chạy",
     "Có thể kiểm tra schema, trọng số cạnh và đối chiếu kết quả ngoại tuyến; số chính thức vẫn ở các bảng kết quả"),
    ("Huấn luyện lại theo chu kỳ", "Danh sách phiên bản, áp dụng và quay lại phiên bản; huấn luyện chạy thủ công",
     "Chưa có ảnh chụp đồ thị gộp dữ liệu mới và lịch tự động"),
]

HINH_MOI = [
    ("fig_v22_4_7_khach_hang.png", "Hình 4.7. Phân hệ khách hàng: danh sách gợi ý cá nhân và lý do gợi ý"),
    ("fig_v22_4_8_cua_hang.png", "Hình 4.8. Phân hệ cửa hàng: hành vi của khách theo thời gian ghi nhận và sản phẩm nên giới thiệu"),
    ("fig_v22_4_9_nha_cung_cap.png", "Hình 4.9. Phân hệ nhà cung cấp giải pháp: đồ thị theo khách và đường lan truyền đến sản phẩm gợi ý"),
]

CHANGELOG = [
    ("Tóm tắt, Summary, 1.3.2, 1.4.2, 3.7.2, 4.5, 4.6, 5.1.3, 5.3.3, 5.4.3, 5.4.4, Bảng 4.12",
     "Mô tả ứng dụng ba phân hệ thay cho hai trang; thêm Hình 4.7 đến 4.9.", "Demo đã làm lại theo D39."),
    ("Bảng 4.9, Hình 4.5, 4.3.3", "Đổi sang λ=0.05.", "Thống nhất với mô hình đề xuất; mục 4.3.3 trước đó mâu thuẫn mục 4.2.4 và 5.1.2."),
    ("4.2.4, 4.3.2, 4.4.1, 4.4.3, 4.6, 5.1.2, 5.3.2, 5.5",
     "Sửa các câu lệch bảng số: tri thức sản phẩm luôn đóng góp lớn hơn, thứ tự mô hình giữ nguyên, cạnh tranh với LightGCN, Coverage thấp.",
     "Đối chiếu lại Bảng 4.5, 4.6 và từng lần chạy."),
    ("4.2.1, 4.3.2, 4.3.3", "Gỡ các câu so với bản v11.", "Người đọc không có bản thảo cũ."),
]


def doan(doc, prefix: str):
    """Mot doan duy nhat theo dau doan, gom ca danh sach."""
    hits = [p for p in doc.paragraphs if p.text.strip().startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"LOI: doan {prefix[:60]!r} khop {len(hits)} lan, can 1")
    return hits[0]


def bang_sau(doc, caption: str) -> Table:
    p = next(x for x in doc.paragraphs if x.text.startswith(caption) and x.style.name != "List Bullet")
    el = p._p.getnext()
    while el is not None and not el.tag.endswith("}tbl"):
        el = el.getnext()
    if el is None:
        raise SystemExit(f"LOI: khong thay bang sau {caption}")
    return Table(el, doc)


def bang_4_9(doc, runs: Path) -> None:
    warm = load_runs(runs, split="test", segment="warm")
    t = bang_sau(doc, "Bảng 4.9.")
    v18.dat_text(t.rows[0].cells[3].paragraphs[0], "BT-DKGRec-GCN (λ=0.05)")
    ten = {"Recall@20": "recall@20", "NDCG@20": "ndcg@20", "HitRate@20": "hit_rate@20"}
    for row in t.rows[1:]:
        cohort, metric = row.cells[0].text.strip(), ten[row.cells[1].text.strip()]
        sub = warm[warm["cohort"] == cohort]
        p = sub[sub["model"].isin(["static_kg_gcn", "bt_dkgrec_l05"])].pivot(index="seed", columns="model", values=metric)
        a, b = p["static_kg_gcn"], p["bt_dkgrec_l05"]
        gia_tri = (f"{a.mean():.6f}", f"{b.mean():.6f}", f"{(b.mean() / a.mean() - 1) * 100:+.2f}%", f"{int((b > a).sum())}/3")
        for cell, text in zip(row.cells[2:], gia_tri):
            v18.dat_text(cell.paragraphs[0], text)


def dong_nhat_so(doc) -> int:
    """Phan tram dau phay, lambda dau cham, trong cac bang ket qua."""
    n = 0
    for cap in ("Bảng 4.5.", "Bảng 4.6.", "Bảng 4.8.", "Bảng 4.9.", "Bảng 4.10."):
        for row in bang_sau(doc, cap).rows:
            for cell in row.cells:
                p = cell.paragraphs[0]
                moi = re.sub(r"([+-]\d+)\.(\d+%)", r"\1,\2", p.text).replace("λ=0,", "λ=0.")
                if moi != p.text:
                    v18.dat_text(p, moi)
                    n += 1
    return n


def them_hinh(doc) -> None:
    """Chen ba anh sau doan gioi thieu o 4.5.3, chep dinh dang anh va chu thich cua Hinh 4.6."""
    from PIL import Image

    cap_goc = next(p for p in doc.paragraphs if p.text.startswith("Hình 4.6.") and p.style.name != "List Bullet")
    anh_goc = Paragraph(cap_goc._p.getprevious(), cap_goc._parent)
    if not anh_goc._p.findall(".//" + qn("a:blip")):
        raise SystemExit("LOI: doan truoc chu thich Hinh 4.6 khong phai anh")
    muc_goc = next(p for p in doc.paragraphs if p.text.startswith("Hình 4.6.") and p.style.name == "List Bullet")
    moc = doan(doc, "Hình 4.7, Hình 4.8 và Hình 4.9 trình bày")._p
    muc = muc_goc._p
    next_id = 9000
    for file, caption in HINH_MOI:
        path = FIG / file
        anh = copy.deepcopy(anh_goc._p)
        blip = anh.find(".//" + qn("a:blip"))
        rid, _ = doc.part.get_or_add_image(io.BytesIO(path.read_bytes()))
        blip.set(qn("r:embed"), rid)
        w, h = Image.open(path).size
        for ext in [*anh.iter(qn("wp:extent")), *anh.iter(qn("a:ext"))]:
            if ext.get("cx"):
                ext.set("cy", str(int(int(ext.get("cx")) * h / w)))
        for tag in (qn("wp:docPr"), qn("pic:cNvPr")):
            for el in anh.iter(tag):
                next_id += 1
                el.set("id", str(next_id))
                el.set("name", file)
        cap = copy.deepcopy(cap_goc._p)
        moc.addnext(anh)
        anh.addnext(cap)
        v18.dat_text(Paragraph(cap, cap_goc._parent), caption)
        moc = cap
        moi = copy.deepcopy(muc)
        muc.addnext(moi)
        v18.dat_text(Paragraph(moi, muc_goc._parent), caption)
        muc = moi


def ly_lich(doc) -> None:
    t = bang_sau(doc, "Thông tin cá nhân")
    for row, value in zip(t.rows[1:4], ("Lê Minh Nguyên", "10/10/1998", "Hồ Chí Minh")):
        if not row.cells[1].text.strip():
            v18.dat_text(row.cells[1].paragraphs[0], value)


def cap_nhat_muc_luc(path: Path) -> int:
    """Ghi so trang tu ban render LibreOffice vao muc luc da luu san; Word van F9 duoc nhu thuong."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", tmp, str(path)], check=True,
                           capture_output=True, timeout=240)
            pdf = Path(tmp) / (path.stem + ".pdf")
            text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True, text=True).stdout
        except (FileNotFoundError, subprocess.SubprocessError) as err:
            print(f"bo qua cap nhat muc luc: {err}")
            return 0
    pages = [re.sub(r"\s+", " ", p) for p in text.split("\f")]
    doc = Document(str(path))
    n, tu_trang = 0, 0
    for hl in doc.element.body.iter(qn("w:hyperlink")):
        if not (hl.get(qn("w:anchor")) or "").startswith(("__RefHeading", "_Toc")):
            continue
        ts = list(hl.iter(qn("w:t")))
        if len(ts) < 2 or not ts[-1].text or not ts[-1].text.strip().isdigit():
            continue
        ten = re.sub(r"^[IVX]+\.\s*", "", re.sub(r"\s+", " ", "".join(t.text or "" for t in ts[:-1])).strip())
        khoa = ten[:45]
        # Trang muc luc co dong cham "....." nen bi bo qua; thu tu tieu de tang dan theo trang.
        for i in range(tu_trang, len(pages)):
            if khoa in pages[i] and "....." not in pages[i]:
                ts[-1].text = str(i + 1)
                tu_trang = i
                n += 1
                break
    doc.save(str(path))
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "De_an_thac_si_v22.docx")
    ap.add_argument("--runs", type=Path, default=ROOT / "experiments" / "runs")
    ap.add_argument("--no-highlight", action="store_true")
    ap.add_argument("--bo-changelog", action="store_true", help="go trang danh muc sua doi -- ban gui co")
    ap.add_argument("--khong-muc-luc", action="store_true", help="khong render de cap nhat so trang")
    args = ap.parse_args()

    v18.TOUCHED.clear()
    doc = Document(str(args.source))
    dem = (len(doc.element.body.findall(f".//{v18.OMML}oMath")), len(doc.tables), len(doc.inline_shapes))
    v18.xoa_to_nen(doc)

    for prefix, moi in THAY_DOAN:
        p = doan(doc, prefix)
        if p._p.findall(f".//{v18.OMML}oMath"):
            raise SystemExit(f"LOI: doan {prefix[:40]!r} co cong thuc")
        v18.dat_text(p, moi)
    for prefix, cu, moi in SUA_CUM:
        p = doan(doc, prefix)
        if cu not in p.text:
            raise SystemExit(f"LOI: khong thay {cu[:50]!r} trong doan {prefix[:40]!r}")
        v18.dat_text(p, p.text.replace(cu, moi))
    for prefix in XOA_DOAN:
        p = doan(doc, prefix)
        p._p.getparent().remove(p._p)
    for prefix in ("Schema Dynamic Knowledge Graph cho Retail Rocket được trình bày", "Quy trình huấn luyện và đánh giá được trình bày"):
        p = doan(doc, prefix)
        moi = re.sub(r" Nguồn sơ đồ nằm tại \S+ và bản PNG tương ứng\.", "", p.text)
        if moi == p.text:
            raise SystemExit(f"LOI: khong thay cau nguon so do trong {prefix[:40]!r}")
        v18.dat_text(p, moi)

    cuoi = next(c for t in doc.tables for r in t.rows for c in r.cells if "năm 2025" in c.text and "Học viên" in c.text)
    for p in cuoi.paragraphs:
        if "năm 2025" in p.text:
            v18.dat_text(p, p.text.replace("năm 2025", "năm 2026"))

    bang_4_9(doc, args.runs)
    v18.dat_text(next(p for p in doc.paragraphs if p.text.startswith("Bảng 4.9.") and p.style.name != "List Bullet"),
                 "Bảng 4.9. BT-DKGRec-GCN (λ=0.05) so với Static KG-GCN, test tại K = 20")
    print(f"dong nhat so: {dong_nhat_so(doc)} o")

    t = bang_sau(doc, "Bảng 4.12.")
    if len(t.rows) != len(BANG_4_12) + 1:
        raise SystemExit("LOI: Bang 4.12 doi so dong")
    for row, values in zip(t.rows[1:], BANG_4_12):
        for cell, text in zip(row.cells, values):
            v18.dat_text(cell.paragraphs[0], text)

    bd.style()
    warm = load_runs(args.runs, split="test", segment="warm")
    v20.thay_anh(doc, "Hình 4.5.", bd.tinh_dong(warm, "fig_v22_4_5_tinh_dong", pair=("static_kg_gcn", "bt_dkgrec_l05")))
    them_hinh(doc)
    ly_lich(doc)
    print(f"dong bo danh muc: {v20.dong_bo_danh_muc(doc)} muc")

    bang = [x for x in doc.tables if x.rows and "Mục" in x.rows[0].cells[0].text]
    if len(bang) != 1:
        raise SystemExit(f"LOI: bang danh muc sua doi khop {len(bang)} lan, can 1")
    for muc, sua, ly_do in CHANGELOG:
        cells = bang[0].add_row().cells
        for cell, text in zip(cells, (f"[v22] {muc}", sua, ly_do)):
            cell.text = text
            v18.TOUCHED.extend(cell.paragraphs)

    sau = (len(doc.element.body.findall(f".//{v18.OMML}oMath")), len(doc.tables), len(doc.inline_shapes))
    if sau != (dem[0], dem[1], dem[2] + len(HINH_MOI)):
        raise SystemExit(f"LOI: so cong thuc / bang / hinh bi doi: {dem} -> {sau}")
    if args.bo_changelog:
        print(f"go trang danh muc sua doi: {v18.xoa_muc_changelog(doc)} phan tu")
    if args.no_highlight:
        v18.xoa_to_nen(doc)
    else:
        print(f"to nen vang {v18.to_nen()} run")
    doc.save(str(args.out))
    if not args.khong_muc_luc:
        print(f"cap nhat so trang muc luc: {cap_nhat_muc_luc(args.out)} muc")
    print(f"cong thuc {sau[0]} · bang {sau[1]} · hinh {sau[2]} -> {args.out}")


if __name__ == "__main__":
    main()
