# Quy ước văn phong khi viết vào đề án

Rút ra bằng cách đối chiếu trực tiếp với `De_an_thac_si_v11.docx`. Bản thảo đầu
tiên của Chương 4 bị viết sai giọng và phải làm lại toàn bộ; tài liệu này tồn
tại để không lặp lại.

## Bảy quy tắc

| # | Quy tắc | v11 | Sai lần đầu |
|:-:|---|---|---|
| 1 | **Giữ thuật ngữ tiếng Anh** | `behavior-time weighting`, `side information`, `split`, `seed`, `validation`, `knowledge graph`, `embedding`, `target`, `Coverage` | dịch hết sang tiếng Việt |
| 2 | **Dấu thập phân là dấu chấm** | `0.021625` | `0,021625` |
| 3 | **Không in đậm giữa đoạn** | không có | in đậm khắp nơi |
| 4 | **Không dùng dấu gạch ngang tu từ** | không có | dùng liên tục |
| 5 | **Câu dài, dày thông tin**, nối bằng dấu chấm phẩy và "do đó", "vì vậy" | xem mẫu dưới | câu ngắn dằn nhịp |
| 6 | **Trình bày, không thuyết phục** — nêu rồi dừng | — | "Cần phân biệt...", "Nói cách khác...", "Đây là hệ quả toán học, không phải quan sát thực nghiệm" |
| 7 | **Kết luận có phòng hộ** | "có thể hữu ích hơn", "đóng góp phù hợp không phải tuyên bố ưu thế phổ quát" | khẳng định dứt khoát |

## Câu mẫu của v11

> Trước khi diễn giải, cần đặt các giá trị tuyệt đối của chỉ số trong ngữ cảnh
> không gian ứng viên. Tập sản phẩm ứng viên gồm 205.106 item từ tập huấn luyện,
> nên một danh sách Top-20 chỉ bao phủ khoảng 0.01% không gian sản phẩm; kỳ vọng
> Recall@20 của một mô hình xếp hạng ngẫu nhiên ở mức xấp xỉ 0.0001.

Một câu, ba mệnh đề, nối bằng dấu chấm phẩy, không nhấn mạnh, không thuyết phục.

## Danh sách gạch đầu dòng

Dạng nhãn rồi giải thích, mỗi mục một dòng ngắn:

> Lịch sử quá thưa: embedding cá nhân hóa không có đủ tín hiệu.
> Target hiếm: addtocart và transaction ít hơn nhiều so với view.

## Định dạng Word

Lấy từ chính `De_an_thac_si_v11.docx`:

| | |
|---|---|
| Font thân bài | Times New Roman 13pt (style `Normal`) |
| Tiêu đề | style `Heading 1` / `Heading 2` / `Heading 3` |
| Bảng | style `Table Grid` |
| Caption bảng | style `Normal`, căn giữa, **in đậm**, đặt **phía trên** bảng |
| Số thứ tự bảng | bám theo v11: Bảng 4.5 và 4.6 là hai bảng kết quả chính |

`scripts/08_make_docx.py` dùng chính file v11 làm template để kế thừa toàn bộ
định nghĩa style, rồi xoá phần thân trước khi ghi nội dung mới. Bản gốc không bị
sửa. Không tự dựng style mới.
