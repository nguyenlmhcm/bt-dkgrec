# Phân tích kết quả — chất liệu viết Chương 4 và 5

Tài liệu này giữ những kết quả phân tích **đã đo được** nhưng **chưa thành quyết định**.
Khác `DECISIONS.md` (ghi quyết định đã chốt), đây là chỗ giữ bằng chứng và lập luận để
đưa vào đề án, cùng những chỗ còn phải xin ý kiến.

Mọi con số dưới đây đo trên 30 run của ma trận λ=0,01 (5 mô hình × 2 cohort × 3 seed),
sinh sau khi sửa phá hòa (D32), schema `curves.csv` mới (D31).

---

## 1. Phép kiểm hiện dùng là SAI phép kiểm — và điều đó đang giấu mất kết quả

### Vấn đề

`CLAUDE.md` quy định: *"So sánh dùng Welch's t-test (hai mẫu độc lập), không dùng paired
test"*, lý do đưa ra là *"sparse ops trên CUDA không bit-exact dù cố định seed"*.

Lập luận đó gộp nhầm hai chuyện:

1. CUDA không bit-exact → không tái lập được một run **chính xác đến từng bit**. Đúng.
2. → seed **không** tạo hiệu ứng chung giữa các mô hình. **Sai.**

### Bằng chứng

Mọi mô hình chạy trên **cùng** bộ seed `{2020, 2021, 2022}` — đây vốn dĩ là **thiết kế
khối ngẫu nhiên đầy đủ** (randomized complete block design). Phân rã phương sai,
cohort Original, split test, phân khúc warm:

| Nguồn phương sai | recall@20 | ndcg@20 |
|---|---:|---:|
| **Do SEED** (chung mọi mô hình) | **73,3%** | **86,4%** |
| Do MÔ HÌNH | 24,1% | 4,7% |
| Phần dư (ngẫu nhiên thật, gồm cả bất định CUDA) | **2,7%** | 9,0% |

Bất định của CUDA nằm gọn trong phần dư 2,7–9,0%. Hiệu ứng seed là 73–86% và **chung
cho mọi mô hình** — dùng phép kiểm độc lập là vứt bỏ nó.

Cơ chế xác nhận điều này:

```
lightgcn         n_nodes = 1.233.091   cạnh tương tác = 1.570.409
static_kg_gcn    n_nodes = 1.262.640   cạnh tương tác = 1.570.409
bt_dkgrec        n_nodes = 1.262.640   cạnh tương tác = 1.570.409
```

Cùng seed → cùng chuỗi negative sampling (cùng 1.570.409 cạnh dương).
`static_kg_gcn` và `bt_dkgrec` có **số node giống hệt** → cùng seed cho **cùng khởi tạo
embedding**. Cặp này ghép cặp là chính xác về mặt cơ học.

### Hệ quả

Original, test, warm, recall@20:

| So sánh | Welch (độc lập) | **Ghép cặp theo seed** |
|---|---:|---:|
| bt_dkgrec vs **lightgcn** | p = 0,2320 | **p = 0,0344** ✅ |
| static_kg_gcn vs lightgcn | p = 0,3952 | p = 0,0617 |
| bt_dkgrec vs static_kg_gcn | p = 0,7160 | p = 0,1453 |

Original, valid, ndcg@20:

| So sánh | Welch | **Ghép cặp** |
|---|---:|---:|
| bt_dkgrec vs **lightgcn** | p = 0,0613 | **p = 0,0033** ✅ |

**Khẳng định chính của đề án — "BT-DKGRec vượt LightGCN" — đạt ý nghĩa thống kê ngay ở
3 seed khi dùng đúng phép kiểm.**

### Ba điều phải nói thẳng khi viết

1. **Ablation vẫn không có ý nghĩa** kể cả khi ghép cặp (p = 0,145 / 0,816 / 0,782 /
   0,484). Hiệu ứng behavior-time thật sự nhỏ.
2. **Bậc tự do = 2.** Ghép cặp với 3 seed rất mong manh. `p = 0,0344` không vững.
3. **Không được đổi phép kiểm sau khi nhìn kết quả rồi im lặng.** Phải báo cáo **cả hai**
   phép kiểm, kèm bảng phân rã phương sai làm căn cứ, và nói rõ vì sao thiết kế là khối.

### Việc còn treo

Chưa sửa `CLAUDE.md`, chưa viết thành quyết định. Cần ý kiến người hướng dẫn.
Script: `scratchpad/blocked.py`.

---

## 2. Vì sao behavior-time yếu — lời giải thích có số đo

### Lập luận

Xếp hạng là **theo từng user**: với user `u`, ta so `z_u · z_j` giữa các item `j`. Nhân
`z_u` với bất kỳ hằng số dương nào **không đổi thứ hạng của user đó**.

Sau chuẩn hóa `Â = D^-½AD^-½`, nếu mọi cạnh của `u` bị nhân cùng hệ số `c` thì `D_u`
cũng nhân `c`, và hàng của `u` chỉ co theo `√c` — một phép **đổi tỷ lệ thuần túy**,
**vô hiệu cho xếp hạng**.

Trọng số chỉ có tác dụng khi nó đổi **tỷ lệ tương đối giữa các item trong hồ sơ của cùng
một user**.

### Dữ liệu RetailRocket đóng sập cánh cửa đó

Phân bố số cạnh tương tác mỗi user, cohort Original (1.027.985 user):

| Số cạnh | Tỷ lệ | Số user |
|---|---:|---:|
| **đúng 1** | **79,6%** | 818.571 |
| đúng 2 | 12,0% | 123.420 |
| từ 3 trở lên | 8,4% | 85.994 |

**Trung vị: 1 cạnh.** Lưu ý: đây là phân bố của **toàn bộ visitor trong train**. Trong
**593 user thực sự được đánh giá**, phân bố khác hẳn — bậc 1 là 43,7%, bậc 2 là 12,8%,
bậc ≥3 là 43,5%.

User một cạnh chỉ có một phần tử khác 0 trên hàng của `Â`, nên behavior-time weighting
không đổi được **thành phần** hàng đó; nó chỉ đổi **hệ số**, và hệ số đó là `√W/√dᵢ`.

> **Đính chính 27/08/2026.** Bản trước kết luận từ đây rằng đổi hệ số là "vô hiệu tuyệt
> đối" với xếp hạng. Sai. Xem mục "Vì sao suy luận cũ sai" bên dưới.

Trong nhóm 8,4% có ≥3 cạnh, đo trực tiếp trên `Â` (mẫu 4.000 user, 341 đủ điều kiện):

| | |
|---|---:|
| Tương quan hàng bt_dkgrec vs static > 0,99 (**chỉ đổi tỷ lệ**) | **62,8%** |
| Mix đổi thật (corr < 0,95) | 11,7% |
| Hệ số tỷ lệ hàng, trung vị | 0,888 |

### Vì sao suy luận cũ sai

Suy luận cũ là: đổi hệ số của hàng `Â` chỉ nhân `z_u` với một hằng số dương, mà nhân hằng
số thì không đổi thứ hạng `z_u · z_j` theo `j`, nên trọng số vô hiệu. Ba lỗ hổng:

1. **Chuẩn hoá không triệt tiêu `W`.** Mẫu số là `√dᵤ` chứ không phải `dᵤ`, nên với user
   bậc 1 hệ số là `√W/√dᵢ` — vẫn phụ thuộc `W`.
2. **Embedding cuối là trung bình qua các lớp, có cả lớp 0.** `h⁰` không nhân với `Â`, nên
   nhân `Â` với một hằng số **không** nhân `z_u` với hằng số đó; hướng của `z_u` đổi.
3. **`Â` đối xứng.** Đổi hàng của user `u` cũng đổi cột `u`, tức đổi `dᵢ` và embedding của
   item, tức đổi thứ hạng của **mọi** user.

Số liệu bác bỏ suy luận cũ: trên Original seed 2020, ở đúng nhóm `warm_deg1` mà suy luận cũ
nói là vô hiệu, `bt_dkgrec` đạt Recall@20 = 0.02490 còn `static_kg_gcn` đạt 0.02104.

### Phần đo vẫn giữ nguyên giá trị

Các số đo trên `Â` ở trên (62,8% hàng có tương quan > 0,99; hệ số tỷ lệ trung vị 0,888) là
**phép đo thật** và vẫn đúng. Điều chúng cho biết là behavior-time weighting **chủ yếu đổi
tỷ lệ chứ ít khi đổi thành phần** hàng — đó là một phát biểu về biên độ, không phải một
phát biểu về việc vô hiệu.

### Câu để viết vào đề án

> Chuẩn hoá đối xứng đưa trọng số cạnh vào lan truyền theo căn bậc hai: tỉ lệ 3:1 giữa
> `transaction` và `view` sau chuẩn hoá còn `√3 ≈ 1,73:1`. Behavior-time weighting vì vậy
> tác động với biên độ nhỏ hơn giá trị danh nghĩa của α; đo trên ma trận kề đã chuẩn hoá,
> phần lớn hàng chỉ đổi tỷ lệ chứ không đổi thành phần lân cận. Điều này giải thích vì sao
> mức chênh lệch quan sát được giữa đồ thị động và đồ thị tĩnh nhỏ, nhưng không triệt tiêu.

Không paper nào trong hai paper nền (LightGCN, KHGT) có phân tích tương tự.

Script: `scratchpad/cancel.py`, `scratchpad/normdiff.py`.

---

## 3. Ta đang KHÔNG chạy cấu hình mà v11 chọn

v11 dòng 849:

> *"Original người dùng có lịch sử dùng 2.024.042 train events và **weighted BPR**. Active
> người dùng có lịch sử dùng BPR chuẩn vì cấu hình không weighted BPR đạt validation
> NDCG@20 cao nhất trong ablation."*

Ma trận 30 run hiện tại dùng **`loss: bpr` ở khắp nơi** (`configs/base.yaml`).

Trong v11, thời gian đi vào **hai** chỗ: trọng số cạnh **và** trọng số từng mẫu trong hàm
loss. Ta đã tắt chỗ thứ hai.

**Vì sao chỗ thứ hai đáng quan tâm:** nhân mẫu huấn luyện với `w` thay đổi mức ảnh hưởng
của nó lên **bảng embedding dùng chung**, và trọng số vào loss **tuyến tính** chứ không bị
căn bậc hai nén như khi vào qua `Â`. Đây là một vị trí khác cho cùng tín hiệu, với biên độ
lớn hơn.

Lưu ý ràng buộc đã chốt: đưa trọng số vào cả hai chỗ làm hai cấu hình khác nhau ở **hai**
biến thay vì một, nên `weighted_bpr` chỉ được dùng trong ablation và bị guard chặn khỏi ma
trận chính.

Code đã có sẵn và đã test: `LOSS_BY_NAME`, `weighted_bpr_loss` (`src/training/loss.py`),
guard `_weighted_bpr_is_ablation_only` (`src/utils/config.py`). Chỉ cần một dòng config.

Chi phí ước tính: 3 run Original ≈ 2,75 giờ.

---

## 4. "Đồ thị động" — vị trí chính xác của đề tài

v11 dòng 418 tự xác định vị trí, và xác định **đúng**:

> *"Nhánh thứ nhất là discrete-time (snapshot-based)... Nhánh thứ hai là continuous-time
> (event-driven)... tiêu biểu là JODIE, DyRep và TGN. **Đề tài này thuộc nhánh
> discrete-time**... Hướng continuous-time... là hướng phát triển tiếp theo (mục 5.4)."*

Định nghĩa hình thức, dòng 549: `G_τ = (V, R, E_τ, A)` với `E_τ = {e_{u,i,b,t} : t ≤ τ}`.

| | Có làm? |
|---|---|
| Đồ thị là hàm của τ (hình thức **và** code — `T_train` là tham số truyền vào) | ✅ |
| Dựng **nhiều** snapshot `G_τ1…G_τT`, học biểu diễn tiến hóa (EvolveGCN, DySAT) | ❌ |
| Biểu diễn cập nhật liên tục theo sự kiện, `z_u(t)` (TGN, JODIE) | ❌ — v11 ghi là hướng phát triển |

**Chính xác: hình thức động, thực thi một lát cắt** (`τ = T_train`).

Câu trả lời cho hội đồng nếu bị hỏi *"đồ thị động ở chỗ nào?"*:

> Đồ thị được tham số hóa theo mốc quan sát τ: `G_τ` chỉ chứa cạnh có `t ≤ τ`, và trọng số
> cạnh là hàm của độ mới so với chính τ đó. Trong thực nghiệm cố định `τ = T_train` để tuân
> thủ quy tắc chống rò rỉ. Trong kiến trúc vận hành ở Chương 3, τ tiến theo thời gian và đồ
> thị được dựng lại theo chu kỳ. Nhánh continuous-time là hướng phát triển ở mục 5.4.

### Ví dụ minh họa (dùng được trong đề án)

Visitor **299** thật, `τ = T_train = 07/08/2015`:

| item | hành vi | ngày | Δt (ngày) | α | exp(−0,01Δt) | w |
|---:|---|---|---:|---:|---:|---:|
| 416219 | view | 26/05 | 73,1 | 1,0 | 0,4815 | 0,4815 |
| 416219 | view | 22/06 | 46,0 | 1,0 | 0,6315 | 0,6315 |
| 149253 | view | 22/06 | 46,0 | 1,0 | 0,6315 | 0,6315 |
| 149253 | addtocart | 22/06 | 46,0 | **2,0** | 0,6315 | **1,2631** |
| 322067 | view | 23/06 | 45,1 | 1,0 | 0,6371 | 0,6371 |

`W(u,i)`: item 149253 → **1,8945**; item 322067 → **0,6371**; item 416219 → **1,1129**.
KG tĩnh gán cả ba bằng **1,0**.

Đổi mốc quan sát — **cùng 5 sự kiện đó**:

| τ | item 149253 | item 322067 | item 416219 |
|---|---:|---:|---:|
| T_train | 1,8945 | 0,6371 | 1,1129 |
| T_train − 30 ngày | 2,5573 | 0,8601 | 1,5023 |
| T_train − 60 ngày | — | — | 0,8773 |

Ở `τ − 60`, hai item **biến mất** (sự kiện của chúng xảy ra sau mốc đó). Đây chính là
`E_τ = {e : t ≤ τ}` thể hiện bằng số.

Script: `scratchpad/example.py`.

### Thí nghiệm rẻ để CHỨNG MINH thay vì khẳng định

Vì `τ` đã là tham số thật trong code, có thể dựng `G_τ` ở nhiều mốc **mà không cần viết
code mới**. Huấn luyện trên `τ = T_train`, `T_train − 15`, `T_train − 30`, đánh giá trên
**cùng** valid/test. Nếu chất lượng giảm khi τ lùi xa → chứng minh độ mới của đồ thị có
giá trị vận hành, tức luận cứ cho kiến trúc dựng lại theo chu kỳ ở Chương 3.

Chi phí: 3 run Original ≈ 2,75 giờ. **Xếp sau Bước 10** — `app/` vẫn còn 0 file.
