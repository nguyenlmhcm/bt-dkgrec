# Tổng quan tài liệu: ngân sách dừng sớm và so sánh công bằng

Ngày lập: 2026-08-25
Loại: scoping review (định hướng, không phải systematic review)
Nguồn: Consensus (Semantic Scholar/Scopus/arXiv), tìm kiếm web, arXiv
Người lập: phiên làm việc BT-DKGRec

## 1. Câu hỏi nghiên cứu

Khi mô hình đề xuất cần nhiều epoch hơn hẳn baseline mới hội tụ, và ngân sách
dừng sớm cố định cắt mất một seed:

- **CH1** — Được phép nâng `patience` chỉ cho mô hình đề xuất không?
- **CH2** — Tài liệu nói gì về ngân sách huấn luyện công bằng giữa các mô hình?
- **CH3** — Được loại một seed hỏng và báo cáo n=2 không?

## 2. Bối cảnh số liệu của đề án

| | Original | Active |
|---|---|---|
| Cấu hình | `patience: 20` × `eval_every: 5` = **100 epoch** | như trên |
| Epoch hội tụ (baseline) | 50–145 | 295–440 |
| Epoch hội tụ (bt_dkgrec_l05) | 85–400 | 670–930 |
| Thời gian 12 run | 11,0 giờ | **2,3 giờ** |

Sự cố: `active/bt_dkgrec_l05/2022` là run **duy nhất trong 24 run** dừng trước
epoch 150 (dừng ở 145, best ở 45).

## 3. Bằng chứng theo chủ đề

### 3.1 "Đặt tham số giống hệt nhau" KHÔNG đồng nghĩa với công bằng

Đây là phát hiện quan trọng nhất và nó đi ngược trực giác thông thường.

Nghiên cứu *Revisiting the Performance of Graph Neural Networks for
Session-based Recommendation* (RecSys '25) chỉ ra: nhiều tác giả cố định kích
thước embedding giống nhau cho mọi mô hình "để so sánh công bằng", nhưng việc
ràng buộc một siêu tham số trọng yếu về cùng một giá trị **có thể tỏ ra bất công
hơn là công bằng** — vì nó bóp nghẹt năng lực mô hình hoá của mô hình này trong
khi không ảnh hưởng mô hình kia. Kết luận của họ: siêu tham số trọng yếu phải
được **tinh chỉnh riêng cho từng mô hình và từng bộ dữ liệu**.

> Lưu ý phạm vi: bài trên nói về *embedding size*. Việc mở rộng lập luận sang
> *patience* là suy luận tương tự của chúng ta, không phải trích dẫn trực tiếp.
> Tuy nhiên logic đồng dạng: patience là ngân sách hội tụ, và mô hình hội tụ
> chậm hơn bị thiệt hại có hệ thống dưới một ngân sách chung.

Shehzad & Jannach (RecSys '23) chứng minh bằng thực nghiệm trên 7 mô hình học
sâu rằng khi baseline không được tinh chỉnh cẩn thận thì **mọi phương pháp đều
có thể được báo cáo là vượt trội** — vấn đề nằm ở phía baseline bị thiệt, chứ
không phải ở phía mô hình đề xuất được ưu ái.

Rendle et al. (2019) cho thấy chỉ riêng việc chạy baseline cho đúng đã đủ khó:
với một matrix factorization được thiết lập cẩn thận, họ vượt qua cả các phương
pháp mới công bố trên MovieLens 10M.

**Hệ quả cho CH1:** nâng patience *chỉ cho mô hình đề xuất* là hướng sai — không
phải vì nó vi phạm nguyên tắc "giống nhau", mà vì nó đặt baseline vào thế bị
thiệt. Nhưng giữ nguyên một patience chung mà mô hình nào cũng bị cắt sớm thì
cũng không phải là công bằng.

### 3.2 Chính sách dừng sớm là một biến gây nhiễu, phải được công bố

Berling, Svahn & Said, *The Hidden Cost of Defaults in Recommender System
Evaluation* (RecSys '25, arXiv:2508.21180) đo trực tiếp việc này: chính sách
dừng sớm **không được ghi trong tài liệu** của RecBole cắt sớm quá trình tìm
kiếm siêu tham số, và mức biến thiên do nó gây ra **ngang bằng với khác biệt
giữa các chiến lược tìm kiếm khác nhau**. Nói cách khác, chính sách dừng có thể
lớn ngang chính hiệu ứng ta đang muốn đo.

Khuyến nghị của họ: coi khung thí nghiệm là **thành phần chủ động của thiết kế
thí nghiệm**, công bố toàn bộ quy trình tinh chỉnh kể cả mặc định của khung, và
làm cho mọi tiêu chí dừng trở nên **hiển thị và cấu hình được**.

### 3.3 Mức patience trong thực tế rất phân tán

| Nguồn | Ngân sách |
|---|---|
| RecBole (`stopping_step`) | mặc định **10** lần eval |
| Giao thức LightGCN thường được nhắc lại | tối đa 1000 epoch, patience **50 epoch**, theo recall@20 |
| Các bài GNN gợi ý khác | 5 / 20 / 50 / 100 epoch — không thống nhất |
| **Đề án này** | **100 epoch** (20 lần eval × 5) |

Không tồn tại một chuẩn thống nhất. Ngân sách của đề án đang ở **mức rộng gấp
đôi giao thức LightGCN thường được trích dẫn**.

> ⚠️ Cần xác minh: con số patience=50 của LightGCN lấy từ mô tả trong các bài
> thứ cấp, chưa đối chiếu trực tiếp bản gốc He et al. (SIGIR 2020). Phải kiểm
> tra bản gốc trước khi trích vào đề án.

### 3.4 Loại bỏ seed là ranh giới đỏ

Tài liệu về *seed hacking* nêu rõ: **chọn seed dựa trên kết quả** là vi phạm
tính toàn vẹn thí nghiệm, và khi cố ý thì bị xếp vào loại gây hiểu lầm. Thực
hành đúng là:

1. **Chọn seed trước khi nhìn kết quả** — cố định và công bố.
2. **Báo cáo phân phối**, không báo cáo điểm đơn lẻ: trung bình, trung vị,
   độ lệch chuẩn, cả giá trị tốt nhất lẫn xấu nhất.
3. Nếu bài toán thật sự không lồi, được phép khởi động nhiều lần — nhưng phải
   **công bố toàn bộ phân phối**, không được giấu.

Về số lượng seed: thực hành khuyến nghị là **10 seed cho so sánh cuối cùng**,
ít hơn chỉ dùng cho giai đoạn thăm dò. Đề án đang dùng 3.

**Hệ quả cho CH3:** không được loại seed 2022 rồi báo cáo n=2. Đây là ranh giới
không thương lượng.

### 3.5 Ghi chú phụ có lợi cho quyết định còn treo về kiểm định ghép cặp

Elliot (Anelli et al., SIGIR '21), khung đánh giá được xây dựng riêng cho tính
nghiêm ngặt, cung cấp sẵn **Wilcoxon và paired t-test** làm phân tích thống kê
mặc định. Ngoài ra, tài liệu về seed hacking ghi nhận rằng kiểm định dựa trên độ
lệch chuẩn có **tỷ lệ phát hiện sai cao hơn paired t-test**.

Đây là bằng chứng ủng hộ hướng ghép cặp theo seed đang treo trong
`PHAN_TICH_KET_QUA.md`, mâu thuẫn với quy tắc hiện hành trong `CLAUDE.md`.
Vẫn cần ý kiến giảng viên hướng dẫn, nhưng đã có chỗ dựa tài liệu.

## 4. Phép kiểm độ nhạy đã thực hiện (không cần train lại)

`curves.csv` lưu validation mỗi lần eval, nên phát lại được luật dừng sớm ở mọi
patience nhỏ hơn mức đã chạy. Script: `scripts/analysis/patience_sensitivity.py`.

| Ngân sách | Số run đổi lựa chọn best_epoch |
|---|---|
| patience=10 (50 epoch) | **11/24** |
| patience=5 (25 epoch) | **14/24** |

Tách theo cohort thì bức tranh rõ hẳn:

- **Original: bền vững.** 10/12 run giữ nguyên best_epoch ở patience=10. Kết quả
  chính của đề án **không phụ thuộc vào ngân sách dừng sớm**.
- **Active: mong manh với MỌI mô hình.** Ở patience=10, `bt_dkgrec/2021` rơi từ
  epoch 295 xuống 65, `static_kg_gcn/2021` từ 305 xuống 40, `lightgcn/2020` từ
  615 xuống 345. Không mô hình nào miễn nhiễm.

**Đây là phát hiện quyết định.** Việc `bt_dkgrec_l05/2022` bị cắt **không phải
đặc tính riêng của λ=0,05**. Đường validation của cohort Active không đơn điệu ở
mọi mô hình; patience=20 tình cờ cứu được 11/12 run và mất 1. Nếu dùng đúng
patience=50 epoch của giao thức LightGCN thì cohort Active sẽ hỏng gần như
toàn bộ.

## 5. Trả lời ba câu hỏi

**CH1 — Không nâng patience cho riêng mô hình đề xuất.** Không phải vì vi phạm
nguyên tắc đối xứng, mà vì nó đặt baseline vào thế bị thiệt — đúng lỗi mà
Shehzad & Jannach chứng minh là làm mọi phương pháp trông như vượt trội.

**CH2 — Ngân sách phải đủ cho mọi mô hình, và phải được công bố.** "Giống hệt
nhau nhưng không đủ" không phải là công bằng. Ngân sách đủ được xác lập bằng
bằng chứng (phép kiểm độ nhạy ở §4), không bằng lời khẳng định.

**CH3 — Không được loại seed.** Ranh giới đỏ. Báo cáo cả 3 seed hoặc chạy lại
cả cohort.

## 6. Khuyến nghị

**Chạy lại toàn bộ cohort Active** — cả 4 mô hình × 3 seed = 12 run — với
patience nâng lên đồng đều (đề xuất 50 lần eval = 250 epoch).

Vì sao đây là phương án đúng chứ không phải phương án đắt:

| | |
|---|---|
| Chi phí | Cohort Active hết **2,3 giờ** cho 12 run. Với patience rộng hơn, ước tính **4–6 giờ**. |
| Tính công bằng | Đồng đều mọi mô hình → không có confound để phản biện. |
| Cohort Original | **Không đụng tới.** §4 đã chứng minh nó bền vững với ngân sách dừng. |
| Biện minh | Không phải "vì kết quả xấu", mà vì phép kiểm độ nhạy cho thấy patience=20 nằm đúng ranh giới đủ/không đủ với cohort này. Lập luận này viết thẳng vào phần phương pháp được. |
| Tính toàn vẹn | Không loại seed nào. |

Phương án này biến điểm yếu nhất của số liệu — một run hỏng — thành một mục
phương pháp luận có bằng chứng.

## 7. Hạn chế của chính bản tổng quan này

- Là scoping review, không có giao thức sàng lọc tái lập được.
- Chỉ 1/3 lượt tìm Consensus dùng được (hết hạn mức tháng); phần còn lại dựa vào
  tìm kiếm web, độ phủ thấp hơn.
- Hai bài RecSys '25 chỉ đọc được tóm tắt/trang arXiv, bản đầy đủ trả 403.
- Con số patience của LightGCN chưa đối chiếu bản gốc (xem §3.3).
- Lập luận mở rộng từ *embedding size* sang *patience* là suy luận tương tự,
  chưa tìm được nguồn nói trực tiếp về patience.

## 8. Tài liệu tham khảo

1. Sun Z. et al. (2020). *Are We Evaluating Rigorously? Benchmarking Recommendation
   for Reproducible Evaluation and Fair Comparison.* RecSys '20.
2. Anelli V.W. et al. (2021). *Elliot: A Comprehensive and Rigorous Framework for
   Reproducible Recommender Systems Evaluation.* SIGIR '21.
3. Shehzad F., Jannach D. (2023). *Everyone's a Winner! On Hyperparameter Tuning
   of Recommendation Models.* RecSys '23.
4. Zhao W.X. et al. (2022). *A Revisiting Study of Appropriate Offline Evaluation
   for Top-N Recommendation Algorithms.* ACM TOIS.
5. Rendle S. et al. (2019). *On the Difficulty of Evaluating Baselines: A Study on
   Recommender Systems.* arXiv.
6. Sun Z. et al. (2022). *DaisyRec 2.0: Benchmarking Recommendation for Rigorous
   Evaluation.* IEEE TPAMI.
7. Berling H., Svahn R., Said A. (2025). *The Hidden Cost of Defaults in
   Recommender System Evaluation.* RecSys '25. arXiv:2508.21180.
8. *Revisiting the Performance of Graph Neural Networks for Session-based
   Recommendation.* RecSys '25. doi:10.1145/3705328.3748156.
9. He X. et al. (2020). *LightGCN.* SIGIR '20. — **cần đối chiếu bản gốc về patience**.
