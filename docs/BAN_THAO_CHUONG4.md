# Bản thảo Chương 4 — các mục viết lại

Bản nháp để đọc và sửa, chưa ghép vào `De_an_thac_si_v11.docx`. Văn phong theo
đúng bản v11: giữ thuật ngữ tiếng Anh, dấu thập phân dùng dấu chấm, câu trình
bày chứ không lập luận.

Số liệu original split lấy từ 18 run trên nhánh `main`. Bản chạy lại trên nhánh
`exp/active-patience50` phải tái lập y hệt vì cùng seed và cùng cấu hình, chỉ bổ
sung phân đoạn theo bậc; nếu lệch thì phải điều tra trước khi dùng. Số liệu
active split là bản mới với patience = 50.

---

## Căn cứ viết lại

Đối chiếu Bảng 4.5 của v11 với kết quả đo lại trên original split, Recall@20:

| Mô hình | v11 | Đo lại |
|---|---:|---:|
| Popularity | 0.011974 | 0.011974 |
| Recent Popularity | 0.013176 | 0.013176 |
| LightGCN | 0.022101 | 0.023736 |
| Static KG-GCN | 0.013668 | 0.027178 |
| BT-DKGRec-GCN | 0.021625 | 0.028778 |

Hai mô hình không tham số trùng khớp đến chữ số cuối. Vì chúng tất định, sự
trùng khớp này xác nhận quy trình tiền xử lý, cách chia theo thời gian và giao
thức đánh giá của bản làm lại giống bản v11; hai bảng số nói về cùng một bài
toán. Ba mô hình có huấn luyện lệch xa hơn, và nguyên nhân nằm trong Bảng 4.4
của v11: epochs = 10 và một seed duy nhất. Bản v11 cũng ghi nhận 10 epoch chưa
đủ để khẳng định mọi mô hình đã hội tụ. Giá trị 0.013668 của Static KG-GCN chỉ
cao hơn Popularity 0.011974 không đáng kể, phù hợp với một lần chạy dừng trước
khi mô hình hội tụ hơn là với giới hạn của kiến trúc.

---

## 4.2.1. Quá trình huấn luyện và cấu hình mô hình

Ma trận thực nghiệm gồm sáu mô hình, ba seed và hai nhóm người dùng. Ba seed
2020, 2021, 2022 được cố định trước khi chạy và lưu trong
`experiments/seeds.json`; không cấu hình nào được chọn lại sau khi quan sát kết
quả test.

**Bảng 4.4 (thay thế). Cấu hình huấn luyện**

| Thành phần | Thiết lập |
|---|---|
| embedding_dim | 64 |
| num_layers | 3 |
| max_epochs | 1000, dừng sớm theo validation NDCG@20 |
| patience | 20 lần đánh giá với original, 50 với active |
| eval_every | 5 epoch |
| batch_size | 65.536 |
| learning_rate | 0.005 |
| reg_weight | 0.0001 |
| loss | BPR chuẩn, một mẫu âm |
| seed | 2020, 2021, 2022 |

Cấu hình này khác bản v11 ở ba điểm, và ba điểm đó giải thích chênh lệch số liệu
giữa hai bản.

Thứ nhất, mô hình được huấn luyện đến khi validation không còn cải thiện thay vì
dừng ở 10 epoch cố định. Trên original split, các mô hình đạt validation NDCG@20
tốt nhất trong khoảng epoch 50 đến 145, tức gấp năm đến mười lăm lần ngân sách
của v11. Một so sánh trong đó mọi mô hình đều chưa hội tụ phản ánh tốc độ hội tụ
trong giai đoạn đầu chứ không phản ánh chất lượng cuối cùng.

Thứ hai, mỗi cấu hình được chạy ba seed thay vì một. Độ lệch chuẩn giữa các seed
trên original split nằm trong khoảng 0.0020 đến 0.0051 ở Recall@20, cùng bậc độ
lớn với khoảng cách giữa các mô hình. Với một seed, không thể tách hiệu ứng của
mô hình khỏi dao động ngẫu nhiên.

Thứ ba, embedding_dim và num_layers lấy theo mục 4.1.2 của LightGCN [He và cộng
sự, SIGIR 2020], trong đó tác giả cố định embedding size bằng 64 cho mọi mô hình
và báo cáo K = 3 cho kết quả tốt nhất trong khoảng 1 đến 4. Mọi mô hình trong ma
trận dùng chung hai giá trị này, nên không mô hình đối chứng nào chạy ở cấu hình
bất lợi so với cấu hình mà tác giả của nó công bố.

Ngân sách dừng sớm được đặt ở cấp split và áp dụng đồng đều cho mọi mô hình
trong cùng split, không đặt riêng cho mô hình đề xuất. Nâng patience riêng cho
mô hình đề xuất sẽ tạo lợi thế không đối xứng; Shehzad và Jannach (RecSys 2023)
cho thấy các so sánh thiếu đối xứng dạng này có thể làm mọi phương pháp đều được
báo cáo là vượt trội. Active split dùng patience = 50 thay vì 20 dựa trên một
phép kiểm độ nhạy: đường validation được phát lại từ `curves.csv` của toàn bộ 24
lần chạy ở các mức patience khác nhau. Trên original, 10 trong 12 lần chạy giữ
nguyên epoch tốt nhất kể cả khi siết patience xuống 10. Trên active, đường
validation không đơn điệu ở mọi mô hình và patience = 20 nằm ở ranh giới đủ.

---

## 4.3.2. So sánh với LightGCN

**Bảng 4.x. BT-DKGRec-GCN so với LightGCN, test tại K = 20, trung bình ± độ lệch
chuẩn trên ba seed**

| Split | Chỉ số | LightGCN | BT-DKGRec-GCN | Chênh |
|---|---|---:|---:|---:|
| original | Recall@20 | 0.023736 ± 0.003524 | 0.028778 ± 0.004961 | +21.24% |
| original | NDCG@20 | 0.014803 ± 0.002754 | 0.015708 ± 0.001940 | +6.11% |
| original | HitRate@20 | 0.046655 ± 0.004244 | 0.055087 ± 0.007604 | +18.07% |
| active | Recall@20 | 0.028601 ± 0.003511 | 0.032445 ± 0.003102 | +13.44% |
| active | NDCG@20 | 0.021816 ± 0.003250 | 0.024814 ± 0.001762 | +13.74% |
| active | HitRate@20 | 0.072650 ± 0.008547 | 0.084046 ± 0.015008 | +15.69% |

BT-DKGRec-GCN cao hơn LightGCN ở cả ba chỉ số xếp hạng trên cả hai split, và cao
hơn ở cả ba seed trong từng phép so sánh.

Kết quả này khác kết luận trong v11, nơi mục 4.3.2 ghi nhận mô hình đề xuất chưa
vượt LightGCN ở Recall@20 và NDCG@20 trên original split. Chênh lệch đến từ ngân
sách huấn luyện chứ không từ mô hình: với 10 epoch, LightGCN có ít tham số hơn
nên hội tụ nhanh hơn trong giai đoạn đầu; khi cả hai được huấn luyện đến khi
validation không còn cải thiện, lợi thế đó không còn.

Với ba seed, kết luận về ý nghĩa thống kê phụ thuộc phép kiểm được chọn, nên đề
án báo cáo cả hai.

| Split | Chỉ số | Welch | Ghép cặp theo seed |
|---|---|---:|---:|
| original | Recall@20 | 0.2320 | 0.0344 |
| active | Recall@20 | 0.2293 | 0.0062 |
| active | NDCG@20 | 0.2524 | 0.1940 |

Kiểm định Welch xử lý hai nhóm như mẫu độc lập. Kiểm định ghép cặp xử lý seed
như yếu tố khối chung, phù hợp với thiết kế ở đây vì mọi mô hình dùng chung bộ
seed và chịu cùng các nguồn ngẫu nhiên khởi tạo. Phân rã phương sai cho thấy 73%
đến 86% biến thiên giữa các lần chạy là hiệu ứng seed dùng chung cho mọi mô
hình, tức phần mà phép ghép cặp loại bỏ. Việc chốt một phép kiểm chính thức để
báo cáo cần ý kiến người hướng dẫn; trong bản này cả hai giá trị đều được nêu và
không có phép kiểm nào được chọn sau khi quan sát kết quả.

---

## 4.3.3. So sánh với Static KG-GCN

Static KG-GCN dùng cùng knowledge graph, cùng category và property, cùng kiến
trúc lan truyền và cùng ngân sách huấn luyện với BT-DKGRec-GCN. Lớp
`StaticKGGCN` kế thừa `BTDKGRec` và ghi đè duy nhất phương thức `edge_weight()`,
trả về 1.0 thay cho `α_b · exp(−λΔt)`. Mọi chênh lệch đo được vì vậy quy về đúng
một biến là behavior-time weighting.

**Bảng 4.y. BT-DKGRec-GCN so với Static KG-GCN, test tại K = 20**

| Split | Chỉ số | Static KG-GCN | BT-DKGRec-GCN | Chênh | Số seed thắng |
|---|---|---:|---:|---:|:---:|
| original | Recall@20 | 0.027178 ± 0.005070 | 0.028778 ± 0.004961 | +5.89% | 3/3 |
| original | NDCG@20 | 0.015909 ± 0.003143 | 0.015708 ± 0.001940 | −1.26% | 2/3 |
| original | HitRate@20 | 0.053963 ± 0.007351 | 0.055087 ± 0.007604 | +2.08% | 2/3 |
| active | Recall@20 | 0.031317 ± 0.003940 | 0.032445 ± 0.003102 | +3.60% | 2/3 |
| active | NDCG@20 | 0.023572 ± 0.002772 | 0.024814 ± 0.001762 | +5.27% | 3/3 |
| active | HitRate@20 | 0.082621 ± 0.019270 | 0.084046 ± 0.015008 | +1.72% | 3/3 |

Không giá trị p nào trong nhóm so sánh này đạt mức 0.05, với Welch trong khoảng
0.55 đến 0.93 và ghép cặp trong khoảng 0.15 đến 0.82. Trên bộ dữ liệu này, đề án
chưa chứng minh được behavior-time weighting cải thiện độ chính xác xếp hạng so
với knowledge graph tĩnh.

Chênh lệch mang dấu dương ở 10 trong 12 phép so sánh giữa hai split, hai split
đánh giá và ba chỉ số; hai trường hợp mang dấu âm có độ lớn −1.26% và −0.00%,
nằm trong dao động giữa các seed. Kết quả vì vậy nhất quán về chiều nhưng có
biên độ nhỏ hơn nhiễu đo, và kết luận phù hợp là chưa đủ bằng chứng để khẳng
định hiệu ứng, không phải bằng chứng cho thấy không có hiệu ứng.

Phần đóng góp của knowledge graph tách riêng khỏi behavior-time weighting thì rõ
hơn. So sánh Static KG-GCN với LightGCN, tức thêm category và property mà không
thêm trọng số thời gian:

| Split | Chỉ số | Chênh | Welch | Ghép cặp | Số seed thắng |
|---|---|---:|---:|---:|:---:|
| original | Recall@20 | +14.50% | 0.3952 | 0.0617 | 3/3 |
| active | Recall@20 | +9.50% | 0.4236 | 0.0278 | 3/3 |

Kết quả này cũng khác kết luận trong v11, nơi mục 4.3.3 ghi nhận Static KG-GCN
thấp hơn LightGCN trên original split. Giá trị 0.013668 của v11 phản ánh một lần
chạy chưa hội tụ.

Hiệu ứng nhỏ của behavior-time weighting có thể giải thích bằng cấu trúc dữ
liệu. Trọng số `W(u,i)` được gộp theo từng cạnh giữa một người dùng và một item,
trong khi phép chuẩn hóa đối xứng `Â = D^(−1/2) A D^(−1/2)` chia hàng của mỗi
người dùng cho tổng trọng số của chính người dùng đó. Với người dùng chỉ có một
cạnh trong graph, trọng số bị chia cho chính nó; vì thứ tự xếp hạng theo từng
người dùng bất biến với phép nhân một hằng số dương, mô hình có behavior-time
weighting và mô hình tĩnh sinh ra cùng một thứ tự cho những người dùng đó.

Phân bố bậc người dùng trong tập huấn luyện của RetailRocket như sau:

| Số cạnh trong graph | Tỷ lệ người dùng |
|---|---:|
| 1 | 79.6% |
| 2 | 12.0% |
| từ 3 trở lên | 8.4% |

Cơ chế behavior-time weighting vì vậy không tác động tới khoảng bốn phần năm số
người dùng trong tập đánh giá, và chỉ số tính gộp trên toàn bộ nhóm warm pha
loãng hiệu ứng khoảng năm lần: một cải thiện 25% trên nhóm chiếm 20.4% sẽ xuất
hiện ở mức khoảng 5% khi tính trên toàn nhóm, cùng bậc độ lớn với các giá trị
quan sát được trong Bảng 4.y.

Quan sát này dẫn tới hai hệ quả. Về thực nghiệm, kết quả cần được báo cáo thêm
theo phân tầng bậc người dùng để đo hiệu ứng trên nhóm mà cơ chế có thể tác
động. Về thiết kế, hướng cải tiến là đưa tín hiệu thời gian vào vị trí không bị
phép chuẩn hóa theo người dùng triệt tiêu, chẳng hạn vào hàm mục tiêu huấn
luyện, hoặc vào cấu trúc graph thông qua nhiều snapshot thời gian thay vì chỉ
qua trọng số cạnh của một snapshot.

Static KG-GCN là mô hình đối chứng tự xây dựng nhằm cô lập một biến. Nó không
phải bản tái lập KGAT hoặc KGCN và không đại diện cho toàn bộ nhóm mô hình gợi ý
dựa trên knowledge graph đã công bố.

---

## Việc còn lại của Chương 4

| Mục | Trạng thái |
|---|---|
| 4.2.2, 4.2.3 bảng kết quả | chờ 12 run original hoàn tất, sinh bằng `make tables` |
| 4.2.4 ablation | ma trận khác v11, phải viết mới |
| 4.4.3 phân tích lỗi | mở rộng từ phần giải thích cơ chế trong 4.3.3 |
| 4.4.4, 5.3.1 hạn chế | ba seed, ablation chưa đạt ý nghĩa, một bộ dữ liệu |
| 4.5 kiểm chứng ứng dụng | viết lại theo ứng dụng đã triển khai |
