# Bản thảo Chương 4 — các mục phải viết lại

Tài liệu này là **bản nháp để đọc và sửa**, chưa ghép vào `De_an_thac_si_v11.docx`.

Số liệu cohort Original lấy từ 18 run đã commit trên nhánh `main`. Bản đang chạy
lại trên nhánh `exp/active-patience50` phải tái lập **y hệt** (cùng seed, cùng
cấu hình, chỉ bổ sung phân đoạn theo bậc). Nếu lệch thì phải điều tra trước khi
dùng. Số liệu cohort Active là bản mới, `patience = 50`.

---

## Vì sao ba mục này phải viết lại

Đối chiếu Bảng 4.5 của v11 với kết quả đo lại, cohort Original, Recall@20:

| Mô hình | v11 | Đo lại | Chênh |
|---|---:|---:|---:|
| Popularity | 0,011974 | 0,011974 | **trùng khít** |
| Recent Popularity | 0,013176 | 0,013176 | **trùng khít** |
| LightGCN | 0,022101 | 0,023736 | +7,4% |
| Static KG-GCN | 0,013668 | 0,027178 | **+98,8%** |
| BT-DKGRec-GCN | 0,021625 | 0,028778 | +33,1% |

Hai mô hình không tham số trùng đến chữ số cuối. Chúng tất định, nên sự trùng
khớp này xác nhận quy trình tiền xử lý, cách chia theo thời gian và giao thức
đánh giá của bản làm lại **giống hệt** bản v11 — hai bảng số nói về cùng một bài
toán, không phải hai thí nghiệm khác nhau.

Ba mô hình có huấn luyện lệch xa, và nguyên nhân nằm trong chính Bảng 4.4 của
v11: `epochs=10`, một seed duy nhất (`seed=2026`). Bản v11 cũng đã tự ghi nhận
*"kết quả hiện tại chưa chứng minh tất cả mô hình đã hội tụ"*. Con số 0,013668
của Static KG-GCN — chỉ nhỉnh hơn Popularity 0,011974 — là dấu hiệu của một lần
chạy dừng trước khi mô hình học xong, không phải giới hạn của kiến trúc.

Do đó hai kết luận của v11 bị đảo chiều sau khi huấn luyện tới hội tụ, và ba mục
dưới đây được viết lại.

---

## 4.2.1. Quá trình huấn luyện và cấu hình mô hình

Toàn bộ ma trận thực nghiệm gồm sáu mô hình × ba hạt giống ngẫu nhiên × hai
nhóm người dùng. Ba hạt giống 2020, 2021, 2022 được cố định trước khi chạy và
lưu trong `experiments/seeds.json`; không có kết quả nào được chọn lại sau khi
nhìn số.

**Bảng 4.4 (thay thế). Cấu hình huấn luyện**

| Thành phần | Thiết lập |
|---|---|
| Kích thước embedding | 64 |
| Số lớp lan truyền | 3 |
| Ngân sách epoch | tối đa 1000, dừng sớm theo `NDCG@20` trên tập xác thực |
| Kiên nhẫn dừng sớm | 20 lần đánh giá (Original), 50 lần (Active) |
| Đánh giá | mỗi 5 epoch |
| Kích thước lô | 65.536 |
| Tốc độ học | 0,005 |
| Hệ số điều chuẩn | 0,0001 |
| Hàm mất mát | BPR chuẩn, một mẫu âm |
| Hạt giống | 2020, 2021, 2022 |

Ba khác biệt so với cấu hình báo cáo trong v11 cần được nêu rõ, vì chúng giải
thích toàn bộ chênh lệch số liệu giữa hai bản:

**Thứ nhất, huấn luyện tới hội tụ thay vì cố định 10 epoch.** Ngân sách 10 epoch
của v11 quá ngắn: trên cohort Original, các mô hình đạt giá trị xác thực tốt
nhất ở epoch 50–145, tức gấp năm đến mười lăm lần ngân sách cũ. Một so sánh
trong đó mọi mô hình đều chưa học xong không cho biết mô hình nào tốt hơn, mà
chỉ cho biết mô hình nào học nhanh hơn trong mười epoch đầu.

**Thứ hai, ba hạt giống thay vì một.** Độ lệch chuẩn giữa các hạt giống trên
Original nằm trong khoảng 0,0020–0,0051 cho `Recall@20`, tức cùng bậc độ lớn với
chính khoảng cách giữa các mô hình. Một lần chạy đơn lẻ không phân biệt được
hiệu ứng của mô hình với dao động ngẫu nhiên.

**Thứ ba, kích thước embedding 64 và ba lớp lan truyền** thay cho 32 và hai lớp.
Hai giá trị này lấy theo mục 4.1.2 của LightGCN [He và cộng sự, SIGIR 2020], nơi
tác giả cố định embedding bằng 64 cho mọi mô hình và báo cáo K = 3 là tốt nhất
trong khoảng 1–4. Mọi mô hình trong ma trận dùng chung hai giá trị này, nên
không mô hình đối chứng nào bị đặt vào cấu hình bất lợi.

Ngân sách dừng sớm được đặt ở **cấp nhóm người dùng**, áp dụng đồng đều cho mọi
mô hình trong cùng nhóm, chứ không đặt riêng cho mô hình đề xuất. Nâng ngân sách
riêng cho mô hình đề xuất sẽ đặt các mô hình đối chứng vào thế bị thiệt — đúng
lỗi mà Shehzad và Jannach (RecSys 2023) chứng minh là làm mọi phương pháp đều có
thể được báo cáo là vượt trội.

Riêng cohort Active dùng kiên nhẫn 50 thay vì 20. Căn cứ là một phép kiểm độ
nhạy: đường xác thực được phát lại từ `curves.csv` của toàn bộ 24 lần chạy với
các mức kiên nhẫn khác nhau. Trên Original, 10 trong 12 lần chạy giữ nguyên
epoch tốt nhất kể cả khi siết kiên nhẫn xuống 10 — ngân sách đã đủ. Trên Active,
đường xác thực không đơn điệu ở **mọi** mô hình, và mức kiên nhẫn 20 nằm đúng
ranh giới đủ/không đủ. Đây là quyết định dựa trên số đo, không phải ước lượng.

---

## 4.3.2. So sánh với LightGCN

**Bảng 4.x. BT-DKGRec-GCN so với LightGCN, kiểm thử tại K = 20, trung bình ± độ
lệch chuẩn trên ba hạt giống**

| Nhóm | Chỉ số | LightGCN | BT-DKGRec-GCN | Chênh |
|---|---|---:|---:|---:|
| Original | Recall@20 | 0,023736 ± 0,003524 | 0,028778 ± 0,004961 | +21,24% |
| Original | NDCG@20 | 0,014803 ± 0,002754 | 0,015708 ± 0,001940 | +6,11% |
| Original | HitRate@20 | 0,046655 ± 0,004244 | 0,055087 ± 0,007604 | +18,07% |
| Active | Recall@20 | 0,028601 ± 0,003511 | 0,032445 ± 0,003102 | +13,44% |
| Active | NDCG@20 | 0,021816 ± 0,003250 | 0,024814 ± 0,001762 | +13,74% |
| Active | HitRate@20 | 0,072650 ± 0,008547 | 0,084046 ± 0,015008 | +15,69% |

BT-DKGRec-GCN cao hơn LightGCN ở cả ba chỉ số xếp hạng, trên cả hai nhóm người
dùng, và **thắng ở cả ba hạt giống** trong từng phép so sánh.

Kết quả này đảo chiều so với v11, nơi mục 4.3.2 kết luận mô hình đề xuất *"chưa
vượt ở Recall@20 và NDCG@20"* trên nhóm người dùng có lịch sử ban đầu. Nguyên
nhân của chênh lệch không nằm ở mô hình mà ở ngân sách huấn luyện: với 10 epoch,
LightGCN — kiến trúc đơn giản hơn, ít tham số hơn — hội tụ nhanh hơn và giành
lợi thế; khi cả hai được huấn luyện tới hội tụ, lợi thế đó biến mất.

**Về ý nghĩa thống kê.** Với ba hạt giống, kết luận phụ thuộc vào việc chọn phép
kiểm, nên báo cáo cả hai:

| Nhóm | Chỉ số | Welch | Ghép cặp theo hạt giống |
|---|---|---:|---:|
| Original | Recall@20 | 0,2320 | **0,0344** |
| Active | Recall@20 | 0,2293 | **0,0062** |
| Active | NDCG@20 | 0,2524 | 0,1940 |

Kiểm định Welch coi hai nhóm là mẫu độc lập. Kiểm định ghép cặp coi hạt giống là
yếu tố khối chung — phù hợp với thiết kế thực nghiệm ở đây, vì mọi mô hình được
huấn luyện trên cùng một bộ hạt giống và chịu cùng các nguồn ngẫu nhiên khởi
tạo. Phân rã phương sai cho thấy 73–86% biến thiên giữa các lần chạy là hiệu ứng
hạt giống dùng chung cho mọi mô hình, tức phần mà phép ghép cặp loại bỏ được.

Đề án báo cáo cả hai giá trị và không chọn một phép kiểm sau khi đã nhìn kết
quả. Việc chốt phép kiểm chính thức cần ý kiến của người hướng dẫn.

---

## 4.3.3. So sánh với Static KG-GCN

Đây là phép so sánh mang tính quyết định của đề tài. Static KG-GCN dùng **cùng**
đồ thị tri thức, **cùng** thông tin danh mục và thuộc tính, **cùng** kiến trúc
lan truyền và **cùng** ngân sách huấn luyện. Nó kế thừa lớp `BTDKGRec` và ghi đè
đúng một phương thức — `edge_weight()` trả về 1,0 thay vì `α_b · exp(−λΔt)`.
Mọi chênh lệch đo được vì vậy quy về đúng một biến: trọng số hành vi–thời gian.

**Bảng 4.y. BT-DKGRec-GCN so với Static KG-GCN, kiểm thử tại K = 20**

| Nhóm | Chỉ số | Static KG-GCN | BT-DKGRec-GCN | Chênh | Thắng |
|---|---|---:|---:|---:|:---:|
| Original | Recall@20 | 0,027178 ± 0,005070 | 0,028778 ± 0,004961 | +5,89% | 3/3 |
| Original | NDCG@20 | 0,015909 ± 0,003143 | 0,015708 ± 0,001940 | **−1,26%** | 2/3 |
| Original | HitRate@20 | 0,053963 ± 0,007351 | 0,055087 ± 0,007604 | +2,08% | 2/3 |
| Active | Recall@20 | 0,031317 ± 0,003940 | 0,032445 ± 0,003102 | +3,60% | 2/3 |
| Active | NDCG@20 | 0,023572 ± 0,002772 | 0,024814 ± 0,001762 | +5,27% | 3/3 |
| Active | HitRate@20 | 0,082621 ± 0,019270 | 0,084046 ± 0,015008 | +1,72% | 3/3 |

Không giá trị p nào trong bảng này đạt mức 0,05, dù dùng Welch (0,55–0,93) hay
ghép cặp (0,15–0,82). Kết luận trung thực là: **trên bộ dữ liệu này, chưa chứng
minh được rằng trọng số hành vi–thời gian cải thiện độ chính xác so với đồ thị
tri thức tĩnh.**

Cần phân biệt phát biểu đó với một phát biểu mạnh hơn mà số liệu **không** ủng
hộ. Chênh lệch mang dấu dương ở 10 trong 12 phép so sánh (hai nhóm × hai split ×
ba chỉ số), và hai trường hợp âm là −1,26% và −0,00%, tức nằm trong nhiễu đo.
Nói cách khác, mô hình động thắng một cách nhất quán nhưng với biên độ nhỏ hơn
dao động giữa các hạt giống. Đây là *chưa chứng minh được có tác dụng*, không
phải *chứng minh được không có tác dụng*.

Ngược lại, phần đóng góp của **bản thân đồ thị tri thức** thì rõ hơn nhiều. So
sánh Static KG-GCN với LightGCN — tức thêm danh mục và thuộc tính sản phẩm mà
không thêm trọng số thời gian:

| Nhóm | Chỉ số | Chênh | Welch | Ghép cặp | Thắng |
|---|---|---:|---:|---:|:---:|
| Original | Recall@20 | +14,50% | 0,3952 | 0,0617 | 3/3 |
| Active | Recall@20 | +9,50% | 0,4236 | **0,0278** | 3/3 |

Kết quả này cũng đảo chiều so với v11, nơi mục 4.3.3 kết luận Static KG-GCN
*"thấp hơn LightGCN"* trên nhóm người dùng có lịch sử ban đầu. Con số 0,013668
của v11 phản ánh một lần chạy chưa hội tụ chứ không phải giới hạn của mô hình.

**Vì sao trọng số hành vi–thời gian có hiệu ứng nhỏ.** Nguyên nhân có thể chỉ ra
bằng cấu trúc dữ liệu chứ không cần suy đoán. Trọng số `W(u,i)` được gộp theo
từng cạnh `(người dùng, sản phẩm)`, còn phép chuẩn hóa đối xứng của LightGCN
chia hàng của mỗi người dùng cho tổng trọng số của **chính người dùng đó**:

```
Â = D^(−1/2) · A · D^(−1/2)
```

Với người dùng chỉ có **một** cạnh, trọng số bị chia cho chính nó và **triệt
tiêu đúng bằng 0**. Xếp hạng theo từng người dùng bất biến với phép nhân một
hằng số, nên với những người dùng đó, mô hình động và mô hình tĩnh sinh ra **kết
quả giống hệt nhau**. Đây là hệ quả toán học, không phải quan sát thực nghiệm.

Trên RetailRocket, phân bố bậc người dùng trong tập huấn luyện như sau:

| Số cạnh | Tỷ lệ người dùng |
|---|---:|
| đúng 1 | **79,6%** |
| đúng 2 | 12,0% |
| từ 3 trở lên | 8,4% |

Nghĩa là cơ chế mà đề tài đề xuất **không thể tác động** tới bốn phần năm số
người dùng trong tập đánh giá. Chỉ số gộp trên toàn bộ nhóm warm vì vậy pha
loãng hiệu ứng khoảng năm lần: một cải thiện thực 25% trên nhóm 20,4% sẽ hiện ra
thành khoảng 5% trên toàn nhóm — đúng bậc độ lớn quan sát được.

Phát hiện này dẫn tới hai hệ quả. Về **thực nghiệm**, kết quả cần được báo cáo
thêm theo phân tầng bậc người dùng, để đo cơ chế ở nơi nó được phép hoạt động
thay vì trên một quần thể mà nó không thể tác động. Về **thiết kế**, nó chỉ ra
một hướng cải tiến cụ thể: đưa tín hiệu thời gian vào chỗ không bị phép chuẩn
hóa theo người dùng triệt tiêu — chẳng hạn vào hàm mất mát, hoặc vào cấu trúc đồ
thị qua nhiều lát cắt thời gian thay vì chỉ vào trọng số cạnh của một lát cắt
duy nhất.

Cuối cùng, cần nhắc lại giới hạn phạm vi đã nêu trong v11: Static KG-GCN là mô
hình đối chứng **tự xây dựng** để cô lập một biến. Nó không phải bản tái lập của
KGAT hay KGCN, và không đại diện cho toàn bộ nhóm mô hình gợi ý dựa trên đồ thị
tri thức đã công bố.

---

## Việc còn lại của Chương 4

| Mục | Trạng thái |
|---|---|
| 4.2.2 / 4.2.3 bảng kết quả | chờ 12 run Original chạy xong, rồi `make tables` |
| 4.2.4 ablation | ma trận khác v11, phải viết mới |
| 4.4.3 phân tích lỗi | dùng phần cơ chế ở 4.3.3, mở rộng thêm |
| 4.4.4 / 5.3.1 hạn chế | n = 3, ablation chưa đạt ý nghĩa, một tập dữ liệu |
| 4.5 kiểm chứng ứng dụng | ứng dụng đã chạy được, viết lại theo bản thật |
