# Việc đang chờ

Cập nhật: 28/08/2026. Hạn nộp: giữa tháng 9/2026.

---

## Trạng thái: train xong, file Word xong

**36/36 lần chạy.** Sáu mô hình × ba seed × hai cohort. Không còn gì phải train.

**Đề án hoàn chỉnh:**

| File | Dùng để |
|---|---|
| `docs/De_an_thac_si_v15.docx` | rà soát — tô vàng 199 chỗ đã sửa |
| `docs/De_an_thac_si_v15_ban_nop.docx` | **nộp** — sạch tô nền |

23 công thức · 47 bảng · 10 hình · không còn số nào của v11 trong thân bài · pytest 319 xanh.

---

## Kết quả cuối cùng — Recall@20

| Bậc | Original (593 user) | Active (234 user) |
|---|---|---|
| Popularity | 0.011974 | 0.006993 |
| Recent Popularity | 0.013176 | 0.003008 |
| LightGCN | 0.023736 | 0.028601 |
| Static KG-GCN | 0.027178 | 0.031317 |
| BT-DKGRec (λ=0.01) | 0.028778 | 0.032445 |
| **BT-DKGRec (λ=0.05)** | **0.030453** | **0.038004** |

Mô hình đề xuất vượt LightGCN **+28,3%** trên Original, **+32,9%** trên Active,
thắng ở **cả ba seed**.

---

## Phần thống kê — đọc kỹ trước khi đụng vào

**Với ba seed, không cặp so sánh nào đạt p < 0,05 hai phía.** Cả Welch lẫn paired.
Gần nhất là LightGCN → BT-DKGRec λ=0.05 trên Original Recall@20: **p = 0,0599**.

Nguyên nhân: độ lệch chuẩn giữa các seed (±0,002 đến ±0,004) **cùng bậc độ lớn** với
khoảng cách giữa các mô hình (+0,001 đến +0,007). LightGCN tự nó dao động
0.019675 → 0.025990, tức 0,0063, gần bằng cả khoảng cách +0,0067 giữa nó và mô hình
đề xuất.

Ba con số hợp lệ đã tính sẵn, dùng được nếu cần:

| | Original | Active |
|---|---|---|
| Một phía, LightGCN → λ=0.05 | **0,0299** | **0,0381** |
| Cận dưới KTC 95% một phía | +0.001325 | +0.001026 |
| Kiểm định dấu, 11/12 bậc dương | **0,0032** | — |

Kiểm định một phía hợp lệ vì thang bậc ở Chương 3 nêu hướng dự đoán **trước khi chạy**.
Nếu dùng, phải ghi **cả hai** giá trị để không bị bắt là giấu.

**Nhưng xem mục "Giọng văn" bên dưới trước khi đưa bất kỳ con số nào vào bài.**

---

## Giọng văn: đây là ĐỀ ÁN ỨNG DỤNG, không phải luận án nghiên cứu

Đối chiếu hai bài mẫu của trường (`đề án Khang - form chuẩn.docx`,
`NGUYENTHIBICHTUYEN.pdf`):

| | Khang | Tuyến |
|---|---|---|
| `p-value` / `p =` | **0 lần** | **0 lần** |
| `độ lệch chuẩn` | 1 | 0 |

**Không đưa bộ máy thống kê vào phần dẫn dắt.** Chương 4 đã sửa theo hướng này:
dẫn bằng sự thật đơn giản ("thắng ở cả ba lần chạy, không lần nào ngược lại"),
bảng kiểm định lùi xuống làm số đối chiếu. Tác giả **không cần nói chữ "p"** khi bảo vệ.

---

## Còn phải làm

1. **Rà v15 bằng mắt trong Word** — mục lục, đánh số bảng/hình có thể cần F9
2. **Đóng port 8501** — `ufw delete allow 8501/tcp`. App Streamlit vẫn mở ra internet,
   không có đăng nhập
3. **Quyết định treo:** chạy ablation tách α khỏi λ (6 run, ~3,5 giờ) hay giữ nguyên
   §4.2.4 như hiện tại

Mục 3 hiện đã xử lý an toàn: §4.2.4 viết lại từ thang bậc thật, và nói thẳng
là α chưa dò. Không chạy thêm vẫn nộp được.

---

## Đã xong, đừng làm lại

| Việc | Nơi lưu |
|---|---|
| Vá Chương 3 (v11 → v12) | `scripts/10_patch_chuong3.py` |
| Vá Chương 1–2 (v12 → v13) | `scripts/11_patch_chuong12.py` |
| Cắt 13 công thức, đánh số lại (v13 → v14) | `scripts/12_cat_cong_thuc.py` |
| Ghép thành đề án hoàn chỉnh (v14 → v15) | `scripts/13_hoan_thien.py` |
| Sinh Chương 4 từ kết quả | `scripts/08_make_docx.py` |
| Sinh 6 hình | `scripts/09_make_figures.py` |
| Trang mạch đề án, 17 câu hỏi bảo vệ | `docs/MACH_DE_AN.html` + artifact |

**Công thức đã đánh số lại ở v14.** Đóng góp riêng nay là **(3.6)–(3.8)**,
không còn là (3.16)–(3.18). Lan truyền (3.11)–(3.14), loss (3.16)–(3.18).

---

## Ba câu trả lời khó, đã chuẩn bị sẵn

**"Sao chọn RetailRocket?"** — chọn theo yêu cầu mô hình: cần nhiều loại hành vi,
timestamp từng sự kiện, thuộc tính sản phẩm, cây danh mục. Độ thưa 3,33% chính là
bài toán, không phải khiếm khuyết.

**"Batch 65.536 ở đâu ra?"** — Codex có codebase v11 gốc và xác nhận: code cũ
**lan truyền lại toàn đồ thị ở mỗi batch**, đã kiểm code dựng lại cũng vậy
(`trainer.py:331` nằm trong vòng lặp batch ở dòng 318). Batch lớn để giảm số lần
lan truyền. Đo được: LightGCN 170 epoch mất 32,1 phút; ở batch 1.024 sẽ là ~34 giờ,
nhân 36 lần chạy thành ~71 ngày. Mặc định model vốn là 2.048; 65.536 do wrapper
Kaggle ghi đè — tức quyết định hạ tầng, không phải quyết định mô hình.

**"Sao số khác bản trước?"** — v11 train 10 epoch nên chưa mô hình nào hội tụ.
Bằng chứng: hai mô hình tất định cho số khớp v11 đến chữ số cuối
(0.006993 / 0.003008 trên Active), chứng tỏ split, mapping và code metric giống hệt;
phần lệch nằm đúng ở các mô hình cần huấn luyện.
