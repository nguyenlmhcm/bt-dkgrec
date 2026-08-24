# Nhật ký quyết định thiết kế

Mỗi mục ghi: quyết định, lý do, và **cần viết gì vào luận văn**. Tài liệu này là
nguồn để soạn phần "khác biệt so với bản v11" khi hoàn thiện đề án.

Ngày lập: 20/8/2026 (Bước 1).

---

## D1. Split theo trục thời gian: cắt theo KHOẢNG THỜI GIAN, không theo số sự kiện

**Quyết định.** `T_train = t_min + int(0.7 × (t_max − t_min))`, biên **bao gồm**
(`ts <= T_train` thuộc train). Tương tự cho mốc 80% của valid.

**Bằng chứng.** Trên `events.csv`:

| | |
|---|---|
| `t_min` | 1.430.622.004.384 |
| `t_max` | 1.442.545.187.788 |
| Khoảng | 11.923.183.404 ms = **138,0 ngày** |
| **`T_train`** | **1.438.968.232.766** |
| **`T_valid_end`** | **1.440.160.551.107** |
| Train events (`ts <= T_train`) | **2.024.042** — khớp Bảng 4.2 của v11, **lệch 0** |
| Train+valid | 2.266.414 (82,23%) |
| Test | 489.687 (17,77%) |

Nếu cắt theo *số sự kiện*, train sẽ có đúng 70% = 1.929.271 dòng, không khớp v11.
Vì cắt theo thời gian nên train chiếm **73,44%** số sự kiện — con số này đúng, không phải bug.

**Viết vào luận văn.** Nói rõ "70/10/20 theo trục thời gian" nghĩa là chia theo
khoảng thời gian, và nêu ba mốc timestamp để người khác tái lập được.

---

## D2. Hàm mất mát: BPR chuẩn cho toàn bộ ma trận thực nghiệm

**Quyết định.** Cả 5 mô hình × 3 seed × 2 cohort dùng **BPR chuẩn (3.30)**.
**Weighted BPR (3.31)** chỉ là **một dòng ablation riêng của `bt_dkgrec`**.
Ràng buộc này được cài cứng: `Config` sẽ báo lỗi nếu đặt `loss: weighted_bpr`
cho bất kỳ mô hình nào khác (`tests/test_config.py::test_weighted_bpr_is_restricted_to_bt_dkgrec_ablation`).

**Lý do.** Bản v11 dùng **weighted BPR cho cohort Original** và **BPR chuẩn cho
cohort Active** (mục 4.2.1). Hệ quả:

1. Hai bảng kết quả (Bảng 4.5 và Bảng 4.6) **không so được với nhau** vì khác hàm mục tiêu.
2. Nếu `bt_dkgrec` dùng weighted BPR còn `static_kg_gcn` dùng BPR chuẩn thì cặp
   ablation **khác hai biến**, phá vỡ lập luận cốt lõi "chỉ khác `edge_weight()`" —
   đây đúng là chỗ hội đồng sẽ chất vấn ở câu hỏi 3.

**Viết vào luận văn.** Ghi rõ: bản v11 dùng loss khác nhau giữa hai cohort khiến hai
bảng không so được với nhau; bản mới thống nhất một loss duy nhất để bảo đảm so sánh
nhất quán, và weighted BPR được giữ lại như một thí nghiệm loại bỏ thành phần.

---

## D3. Lọc PropertyValue: `freq >= 5` trước, rồi lấy top 50.000 theo tần suất

**Quyết định.** Hai cơ chế lọc trong hai tài liệu được áp dụng **nối tiếp**:
1. Bỏ `property ∈ {categoryid, available}` (KG_DESIGN §2.4)
2. Bỏ PV có tần suất `< min_pv_freq = 5`
3. Giữ **top `max_property_nodes = 50.000`** PV theo tần suất giảm dần

**Đổi tên tham số.** v11 (Bảng 4.4) gọi tham số này là `max_property_edges`, nhưng
ngữ nghĩa thật là **số node PropertyValue**, không phải số cạnh — chính v11 cũng
phải chú thích lại điều này. Code dùng tên đúng bản chất là `max_property_nodes`.

| Tên trong v11 | Tên trong code | Ngữ nghĩa |
|---|---|---|
| `max_property_edges` | `max_property_nodes` | Số node PropertyValue được giữ lại |

**Cảnh báo về con số 3.307.294.** Con số cạnh item-property của v11 chỉ dùng để
**chọn cách diễn giải spec**, tuyệt đối không dùng để ép kết quả. Nếu kết quả mới ra
3,1M hay 3,5M cạnh thì **giữ nguyên số mới** và ghi vào `graph_stats.json`.
Chỉ dừng lại tìm bug nếu lệch **một bậc độ lớn**.

**Viết vào luận văn.** Nêu rõ tên và ngữ nghĩa tham số để tránh hiểu nhầm số cạnh.

---

## D4. `edge_weight()` do LỚP MODEL quyết định, không bao giờ do config

**Quyết định.** Config chỉ chứa **tham số** của công thức (`alpha`, `lambda_decay`,
`d_day`). **Không** có công tắc kiểu `weighting.enabled` trong YAML.

**Lý do.** Nếu vừa có công tắc config vừa có override lớp thì tồn tại **hai đường**
để tắt behavior-time weighting → có thể vô tình chạy `bt_dkgrec` với weighting đã tắt
mà bảng kết quả vẫn ghi là `bt_dkgrec`. Đây là loại lỗi âm thầm làm hỏng đúng cặp
so sánh quan trọng nhất của luận văn.

```
BTDKGRec.edge_weight(b, dt)      -> alpha[b] * exp(-lambda * dt)     # (3.17)
StaticKGGCN(BTDKGRec).edge_weight -> 1.0                             # override duy nhat
LightGCN.edge_weight              -> 1.0  + khong dung side info
```

**Ràng buộc phải giữ khi làm Bước 7.** `git diff` giữa `bt_dkgrec.py` và
`static_kg_gcn.py` chỉ được khác đúng hàm `edge_weight()`.

---

## D5. Coverage@K — định nghĩa chốt

```
Coverage@K = |{ item phân biệt xuất hiện trong Top-K của TOÀN BỘ user được đánh giá }|
             ------------------------------------------------------------------------
                                      |I_train|
```

Định nghĩa này được ghi vào docstring của `src/evaluation/metrics.py` (Bước 5).

**Kiểm chứng ngược từ v11.** LightGCN có Coverage@20 = 0,013227 trên cohort Original
với `|I_train|` = 205.106 → 0,013227 × 205.106 ≈ **2.713 item phân biệt**, trong khi
tổng số ô Top-K là 593 × 20 = 11.860. Hợp lý → cách diễn giải đúng.

---

## D6. Tham số chưa được đặc tả ở tài liệu nào — giá trị đã chốt

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `recent_popularity.recent_window_days` | **14** | Đã chọn theo **valid** NDCG@20 — xem D19 |
| `training.monitor` | `ndcg@20` trên **valid** | Không bao giờ đọc test (quy tắc 7) |
| `training.patience` | 20 lần eval | |
| `training.eval_every` | 5 epoch | |
| `training.max_epochs` | **1000 cho MỌI mô hình** | Xem D7 |
| `training.num_negatives` | 1 | |
| Định nghĩa 1 epoch | 1 lượt duyệt hết cạnh tương tác train | |

---

## D7. Ngân sách huấn luyện: cap 1000 epoch cho mọi mô hình

**Quyết định.** `max_epochs = 1000` **giống nhau cho cả 5 mô hình**, điểm dừng thật
do early stopping trên valid quyết định.

**Lý do.** CLAUDE.md yêu cầu `max_epochs >= 300`, nhưng LightGCN gốc train 1000 epoch.
Đặt cap bằng đúng ngân sách của tác giả baseline thì không ai chất vấn được
"baseline bị bóp ngân sách". Early stopping đã chặn overfit nên cap cao không gây hại.
Đây là cách sửa lỗi `epochs: 10` của bản v11 (mọi mô hình bị dừng ở 10 epoch,
LightGCN chưa hội tụ → so sánh không hợp lệ).

**Bằng chứng bắt buộc.** Mỗi run phải sinh `curves.csv`; khi bảo vệ phải chỉ ra được
đường valid của LightGCN đã plateau.

---

## D8. `popularity` / `recent_popularity` có std = 0 — phải chú thích ở BẢNG

**Quyết định.** Hai mô hình này tất định nên chạy 3 seed vẫn cho **std = 0**.
Vẫn chạy đủ 3 seed để giữ đúng quy tắc "mọi mô hình cùng bộ seed".

**Bắt buộc.** Ghi chú "std = 0 do mô hình tất định, không phụ thuộc seed" vào
**chú thích dưới bảng kết quả** trong luận văn — không chỉ để trong log — để hội đồng
không hiểu nhầm là lỗi tính toán hoặc lỗi ghi số.

---

## D9. Đọc file lớn: pandas + chunksize (không thêm DuckDB/Polars)

**Quyết định.** VPS chỉ có **3 GB RAM**, `item_properties` có **20.275.902 dòng / 852 MB**.
Dùng `pandas.read_csv(..., chunksize=1_000_000)` với:
- `usecols` — chỉ đọc cột cần
- `dtype` khai báo tường minh (`int32` / `category`)
- **lọc ngay trong từng chunk** (bỏ `available`, cắt `timestamp <= T_train`) trước khi gộp

**Lý do.** Giữ đúng chuẩn công nghệ đã chốt (PyTorch + pandas + scipy.sparse), không
thêm dependency lạ phải giải thích thêm trong luận văn.

---

## D10. Tách `requirements.txt` và `requirements-train.txt`

**Quyết định.** `requirements.txt` chứa phụ thuộc lõi (numpy/pandas/scipy/pyarrow/
pydantic/pytest) — chạy được trên VPS không GPU. `requirements-train.txt` thêm `torch`
— cài trên Colab.

**Lý do.** VPS 3 GB RAM / 13 GB đĩa không train được; cài torch trên VPS chỉ tốn tài
nguyên. Việc tách này giữ cho `make setup` và `make test` chạy nhanh trên VPS.

---

## D11. Bằng chứng: bản v11 nhiều khả năng **không áp mốc `T_train`** cho `item_properties`

**Bối cảnh.** Số cạnh item-property mới ra 2.922.197, thấp hơn v11 (3.307.294) **−11,6%**.
Em truy nguyên bằng cách chạy lại đúng pipeline với **một** thay đổi duy nhất: bỏ điều
kiện `timestamp <= T_train`.

| Cách dựng | Cặp (item, property) | PV node | Cạnh item-property | So với v11 |
|---|---|---|---|---|
| **Có áp `T_train`** (đúng quy tắc 3) | 3.946.253 | 28.237 | **2.922.197** | −11,6% |
| Bỏ mốc thời gian (rò rỉ) | 4.414.856 | 30.187 | **3.344.887** | **+1,1%** |

**Diễn giải.** Đây là bằng chứng mạnh — không phải chứng minh tuyệt đối — rằng bản v11
đã nạp thuộc tính sản phẩm ở **mọi** thời điểm, kể cả sau `T_train`, tức là đồ thị huấn
luyện đã nhìn thấy trạng thái tương lai của sản phẩm. Đó chính là **vi phạm quy tắc
chống rò rỉ số 3**.

**Quyết định.** Giữ cách đúng (áp mốc). Chấp nhận số cạnh thấp hơn v11.

**Viết vào luận văn.** Đây là một luận điểm có lợi: bản mới chặt chẽ hơn về mặt giao thức.
Nêu rõ số cạnh giảm ~11,6% là **hệ quả của việc siết quy tắc chống rò rỉ**, không phải
mất mát thông tin do lỗi kỹ thuật.

---

## D12. Tần suất PropertyValue tính trên **item trong train**, sau khi đã áp quy tắc 4

**Quyết định.** `freq(pv)` = số **item trong train** mang PV đó, tính trên bảng đã lấy
bản ghi mới nhất ≤ `T_train` cho mỗi cặp (item, property).

**Đo thực tế hai cách diễn giải:**

| Phạm vi tính tần suất | PV có freq ≥ 5 | Cap 50.000 | PV node giữ lại | Cạnh item-property |
|---|---|---|---|---|
| **Item trong train (đã chọn)** | 28.237 | không ràng buộc | 28.237 | 2.922.197 |
| Toàn bộ item | 71.060 | có ràng buộc | 50.000 | 2.948.778 |

**Lý do chọn.** Hai cách cho số cạnh gần như nhau (lệch 0,9%), nhưng cách thứ hai thêm
21.763 PV node mà chỉ đóng góp 26.581 cạnh — tức bậc trung bình ~1,2, gần như node cô lập,
không có ích cho lan truyền mà còn làm loãng đồ thị. Cách đã chọn cho `freq` đúng nghĩa
**bậc của node PV trong đồ thị thật**.

**Hệ quả cần công khai.** Với cách này, tham số `max_property_nodes = 50.000`
**không bao giờ ràng buộc** (chỉ có 28.237 PV vượt ngưỡng tần suất). Phải ghi rõ điều
này trong luận văn thay vì để người đọc tưởng cap đang có tác dụng.

---

## D13. Dòng "tổng entity" của v11 **không tính node Visitor**

Bảng 4.2 của v11 ghi "Entity count của graph học được" = 214.396 (original).
Đối chiếu ngược: 214.396 − 205.106 item ≈ 9.290, tức chỉ còn chỗ cho category + PV.
Vậy con số này **không bao gồm 1.027.985 Visitor**.

Audit mới in **cả hai dòng** để không so nhầm định nghĩa:
- `entity (item+category+PV)` — so được với v11
- `tổng node kể cả visitor` — con số thật của đồ thị huấn luyện

Lưu ý: con số entity của v11 giữa hai cohort không nhất quán với nhau dưới bất kỳ định
nghĩa đơn nhất nào (original hàm ý ~8.000 PV, active hàm ý ~43.000 PV — ngược chiều trực
giác), nên dòng này chỉ dùng tham khảo lỏng.

---

## D14. Guard chạy như GATE, không phải test-only — và không có đường tắt

**Quyết định.** `run_preprocess_guards()` được gọi **hai lần** trong mỗi lần
`01_preprocess.py` chạy:
1. trên cấu trúc còn trong bộ nhớ (ngay sau khi dựng)
2. **đọc lại từ file Parquet đã ghi**

**Lý do có lượt thứ hai.** Lượt này bắt được lỗi ở khâu ghi/đọc mà lượt in-memory
không thấy. Đúng lúc làm Bước 3 nó có ích ngay: `side_item_property.parquet` đang lưu
**item_id thô** trong cột tên `item_idx` (giá trị lớn nhất 466.864 trong khi mapping chỉ
có 205.106 item). Số liệu audit không đổi (đếm `nunique` giống nhau) nên bảng audit của
Bước 2 vẫn đúng, nhưng bước dựng ma trận kề ở Bước 4 sẽ địa chỉ sai hàng. Đã sửa và bổ
sung guard `assert_index_within_mapping` để lỗi loại này không tái diễn.

**Không có cờ `--skip-guards`.** Cố tình không cài, để không tồn tại cách nào chạy
pipeline vòng qua guard.

**Rule 5 và rule 6 KHÔNG nằm trong gate tiền xử lý.** Ở thời điểm tiền xử lý chưa có
candidate set lẫn negative sampler; nếu vẫn gọi thì assertion trở thành
`I_train ⊆ I_train` — luôn đúng, in ra PASS mà không kiểm chứng gì, tạo cảm giác an toàn
giả. Hai rule này có hàm + test đầy đủ, sẽ gắn vào đúng nơi có artefact:
evaluator (Bước 5) và sampler (Bước 6).

---

## D15. `popularity` xếp hạng theo **hành vi mục tiêu**, không phải theo mọi tương tác

**Quyết định.** Điểm phổ biến = số sự kiện `addtocart OR transaction` của item trong
train. Cấu hình `popularity_signal: target` (đặt `all` để kiểm tra độ nhạy).

**Lý do.** Bài toán là dự báo **hành vi mục tiêu**. Nếu mốc sàn xếp hạng theo tổng lượt
tương tác (chủ yếu là view) thì nó đang trả lời một câu hỏi khác, và so sánh sẽ không
cùng đơn vị. Mốc sàn phải mạnh nhất có thể trong phạm vi "không cá nhân hoá" — mốc sàn
yếu giả tạo sẽ thổi phồng đóng góp của mô hình đề xuất.

**Kiểm chứng.** Trên cohort Active, cách này tái lập **chính xác** ba trên bốn chỉ số của
v11 (xem D16), nên nhiều khả năng v11 cũng dùng tín hiệu này.

---

## D16. Cohort `original` KHÔNG lọc visitor — nếu không sẽ không còn cold user để báo cáo

**Lỗi đã sửa.** Bước 2 gọi `apply_cohort()` cho **cả hai** cohort. Với `original`
(`min_active_events = 0`), tập visitor lấy từ `value_counts` của train nên chỉ gồm người
**có mặt trong train** — kéo theo mọi sự kiện valid/test của **cold user bị xoá sạch**.
Hệ quả: phân đoạn cold luôn rỗng, trong khi CLAUDE.md yêu cầu báo cáo riêng phân đoạn này.

**Sau khi sửa** (chỉ lọc khi `min_active_events > 0`):

| | valid | test |
|---|---|---|
| warm | 552 | 593 |
| **cold** | **3.136** | **6.786** |

Bảng audit khối B vẫn **lệch 0** so với v11 vì cold user theo định nghĩa không có sự kiện
train, nên không đụng tới bất kỳ con số train nào.

**Cohort `active` có cold = 0 theo cấu tạo** — mọi user được đánh giá đều phải có ≥5 sự
kiện train nên đều là warm. Đây là tính chất, không phải lỗi; cần ghi chú khi trình bày bảng.

---

## D17. `evaluation.batch_size = 64` — đã từng làm OOM ở 256

**Sự cố.** Chạy `popularity` trên cohort original bị kernel giết:
`Out of memory: Killed process (python) anon-rss:1.317GB`.

**Nguyên nhân.** `np.argpartition` trả về mảng chỉ số **int64**:

| Thành phần | batch 256 × 205.106 item |
|---|---|
| `scores` float32 | 210 MB |
| `-scores` (bản sao do phép phủ định) | 210 MB |
| `argpartition` int64 | **420 MB** |
| tổng | **~840 MB** |

**Sửa hai chỗ.** (1) Phân hoạch trực tiếp trên `scores` với `kth = n - K` thay vì trên
`-scores`, bỏ hẳn một bản sao. (2) Hạ `batch_size` xuống 64 → ~160 MB. Đo lại: RAM đỉnh
**0,73 GB**, chạy 214 giây cho cohort original.

**Ghi chú vận hành.** Lỗi này chỉ lộ ra vì chạy trên VPS 3 GB. Trên Colab 12 GB nó sẽ ẩn
đi rồi bùng lên ở cấu hình lớn hơn.

---

## D18. Item đã bị lọc không bao giờ được lọt vào Top-K

Điểm của item đã xem bị đặt `-inf`, nhưng nếu số ứng viên ít hơn K thì phép sắp xếp vẫn
trả về đủ danh sách **kể cả các ô `-inf`**. Đã sửa: mọi vị trí có điểm không hữu hạn bị
đổi thành `-1` (ô trống), và `coverage_at_k` bỏ qua ô `-1` để không thổi phồng độ phủ.

Trên dữ liệu thật (205.106 item, K = 20) nhánh này không kích hoạt, nhưng nó sẽ kích hoạt
ở bất kỳ thí nghiệm nào có tập ứng viên nhỏ — và khi đó sẽ sai âm thầm.

---

## D19. `recent_window_days = 14` — chọn trên valid, cả hai cohort đều đồng thuận

**Cách chọn.** Huấn luyện `recent_popularity` với cửa sổ ∈ {7, 14, 30} ngày, chấm điểm
trên **valid warm**, chọn theo NDCG@20. **Không bao giờ đọc test** (quy tắc 7).

| Cửa sổ | original — valid NDCG@20 | active — valid NDCG@20 |
|---|---|---|
| 7 ngày | 0,020521 | 0,007678 |
| **14 ngày** | **0,021333** ★ | **0,007735** ★ |
| 30 ngày | 0,020644 | 0,007707 |

**Vì sao quan trọng.** Hai cohort chọn cùng một giá trị nên dùng chung `14` — giao thức
đồng nhất giữa hai bảng kết quả. Nếu hai cohort chọn khác nhau thì phải công khai rằng
baseline được cấu hình riêng cho từng cohort, và điều đó làm hai bảng khó so với nhau
(đúng loại vấn đề mà D2 đã phải xử lý với hàm mất mát).

**Ghi chú.** Chênh lệch giữa ba cửa sổ rất nhỏ (~4%), nên baseline này không nhạy với
tham số. Đó là tin tốt: kết quả của nó là mốc sàn ổn định, không phải sản phẩm của việc
dò tham số.

---

## D20. Đổi phân công hạ tầng: **mọi thực nghiệm chạy trên Colab**, kể cả mốc sàn

**Quyết định (21/8/2026).** Phân công lại:

| | Việc |
|---|---|
| VPS | Viết code, cấu trúc dự án, notebook, dựng KG phục vụ demo, đẩy lên GitHub |
| **Colab** | **Chạy toàn bộ thực nghiệm — mọi mô hình, mọi seed, mọi cohort** |

Trước đó `popularity` và `recent_popularity` chạy trên VPS vì chúng không cần GPU.

**Lý do kỹ thuật.** VPS có 3,9 GB RAM nhưng VS Code server và các tiến trình nền chiếm
~2,6 GB, chỉ còn **343–768 MB** cho pipeline. Hai run `original/recent_popularity` bị
kernel giết (`exit 137`) ở mức rss chỉ 370–477 MB. Đã hạ `evaluation.batch_size` từ 256 →
64 → 16 nhưng biên an toàn vẫn quá mỏng và phụ thuộc vào việc lúc đó máy đang rảnh hay bận.

**Lý do học thuật — quan trọng hơn.** Chạy một phần thực nghiệm trên VPS và phần còn lại
trên Colab tạo ra bảng kết quả **trộn hai môi trường**. Với mô hình tất định thì con số
không đổi, nhưng nó phá vỡ nguyên tắc "mọi mô hình dùng chung giao thức" và tạo ra một câu
hỏi không cần thiết khi bảo vệ. Một môi trường duy nhất thì không có gì để hỏi.

**Hệ quả đã thực hiện.** 10 run đã chạy trên VPS bị **loại khỏi** `experiments/runs/`
(chuyển ra ngoài, không xoá). Kết quả chính thức sẽ được sinh lại toàn bộ trên Colab.

**`evaluation.batch_size`.** Giữ 16 trong `configs/base.yaml` để VPS vẫn chạy được khi cần
thử nhanh; notebook Colab ghi đè bằng `--eval-batch-size 512`. Đây là tham số hạ tầng, không
ảnh hưởng kết quả — chỉ đổi số user chấm điểm mỗi lô.

---

## D21. Notebook Colab phải chống được việc bị ngắt giữa chừng

**Vấn đề.** Colab dừng đột ngột là chuyện bình thường (hết phiên, mất kết nối, hết hạn mức
GPU). Nếu notebook chỉ lưu kết quả ở cuối thì mất sạch, phải chạy lại từ đầu.

**Ba lớp checkpoint, tất cả đặt trên Drive:**

| Giai đoạn | Lưu ở | Khi chạy lại |
|---|---|---|
| Tiền xử lý (`data/interim`) | `cache/interim` | Nạp lại, bỏ qua bước tính |
| Đồ thị (`data/processed`) | `cache/processed` | Nạp lại, bỏ qua bước tính |
| Từng run | `runs/` | Run đã xong thì bỏ qua |

**Cache gắn với commit.** Mỗi thư mục cache đi kèm một file `<tên>.commit` ghi commit đã
sinh ra nó. Notebook chỉ dùng lại cache khi commit khớp với commit đang chạy. Đổi code →
commit đổi → cache tự hết hiệu lực, tính lại.

Đây là điểm quan trọng về mặt liêm chính, không chỉ tiện lợi: **không bao giờ có chuyện
kết quả được sinh từ dữ liệu trung gian của một phiên bản code khác** mà không ai biết.

**Run được đồng bộ NGAY sau khi xong**, không đợi đến cuối notebook. Colab chết giữa chừng
thì những run đã hoàn thành vẫn còn nguyên trên Drive.

---

## D22. Chuẩn trình bày biểu đồ cho luận văn

Toàn bộ hình do `src/evaluation/figures.py` sinh ra, không vẽ rời rạc trong notebook — để
mọi hình trong luận văn nhất quán và để `scripts/06_make_tables.py` (Bước 9) dùng lại đúng
mã đó.

**Màu gắn với mô hình, không gắn với thứ hạng.** `MODEL_COLORS` cố định: một hình chỉ có ba
mô hình vẫn giữ nguyên màu của từng mô hình. Nếu màu chạy theo thứ hạng thì hai hình cạnh
nhau sẽ có cùng màu chỉ đôi ý nghĩa — người đọc so sánh sai mà không biết.

| Mô hình | Màu | Hoa văn |
|---|---|---|
| Popularity | `#2a78d6` xanh dương | (đặc) |
| Recent Popularity | `#eb6834` cam | `//` |
| LightGCN | `#1baf7a` xanh ngọc | `\\` |
| Static KG-GCN | `#4a3aa7` tím | `xx` |
| BT-DKGRec-GCN | `#e34948` đỏ | `..` |

**Đã kiểm định mù màu** (không phải ước lượng bằng mắt): CVD ΔE cặp liền kề xấu nhất **9,2**
(ngưỡng ≥ 8), normal-vision ΔE **27,6** (sàn ≥ 15), dải độ sáng và sàn chroma đều đạt. Một
cảnh báo tương phản ở màu xanh ngọc được giải bằng **nhãn số in trực tiếp trên cột** — vốn
cũng là thứ luận văn cần.

**Hoa văn + nhãn số = mã hoá thứ cấp.** Luận văn in đen trắng hoặc photocopy vẫn phân biệt
được mô hình, không phụ thuộc màu.

**Không bao giờ dùng hai trục y.** Hai đại lượng khác thang thì tách thành hai panel.

**Định dạng số thích nghi.** Coverage ~2,5e-4; nếu in cứng 4 chữ số thập phân thì 0,000249
và 0,000268 đều thành "0.0002" — mất thông tin và tạo cảm giác sai. Dùng 3 chữ số có nghĩa.

**Xuất cả PNG 300 dpi và PDF vector** — Word dùng PNG, LaTeX dùng PDF.

---

## D23. Chốt hạ tầng: Colab train qua GitHub, artifact quay về theo hai đường

**Luồng.** VPS viết code → `git push` → Colab `git pull` → train GPU → artifact quay về.
VPS **không train**, kể cả mốc sàn.

### Hoàn nguyên mọi chỉnh tạm cho VPS

| Tham số | Từng bị hạ | Nay | Ghi chú |
|---|---|---|---|
| `training.batch_size` | (không đụng) | **65536** | Đúng chuẩn CLAUDE.md từ đầu |
| `evaluation.batch_size` | 256 → 64 → 16 | **256** | Đã hoàn nguyên |
| `training.device` | ghi chú "VPS smoke test cpu" | **cuda** | Bỏ ghi chú |

`configs/`, `src/`, `scripts/`, `Makefile` đã quét sạch, không còn chuỗi "VPS" nào.

### Artifact quay về theo hai đường

| File | Về đâu | Lý do |
|---|---|---|
| `metrics.json`, `curves.csv`, `config.yaml`, `seed.txt` | **Git** | Nhẹ; cần version control; Bước 9 `make tables` đọc **toàn bộ** run từ repo |
| `topk.csv`, `train.log` | **Drive** | Nặng; `topk.csv` chỉ demo Bước 11 cần đọc, `train.log` chỉ dùng khi chẩn đoán |

`train.log` không nằm trong bảng phân loại ban đầu. Xếp về Drive vì với `max_epochs = 1000`
nó sẽ phình to, mà nội dung chỉ hữu ích khi chẩn đoán — bằng chứng hội tụ nằm ở `curves.csv`,
và file đó về Git.

`.gitignore` chỉ chặn đúng hai mẫu `experiments/runs/**/topk.csv` và
`experiments/runs/**/train.log`, thay cho mẫu chặn toàn bộ `experiments/runs/*` trước đây.
Đã kiểm chứng bằng `git check-ignore` cho cả sáu file.

**Commit ngược sau MỖI run**, không gom cuối buổi. Colab timeout giữa chừng thì phần đã chạy
vẫn nằm an toàn trên GitHub.

### Trạng thái tiến độ nằm trong chính repo

Một run coi là xong khi `metrics.json` của nó có mặt trong repo. Nên mở phiên Colab mới,
`git pull`, là chạy tiếp đúng chỗ dừng — kể cả trên máy khác. Không cần file trạng thái riêng,
không có chuyện trạng thái lệch với thực tế.

### Thông tin đăng nhập

Token đọc theo thứ tự: **Colab Secrets** (`GITHUB_TOKEN`, khuyến nghị) → `getpass` nhập tay.

Ba ràng buộc được giữ trong code:
- Token **không nằm trong notebook**, không nằm trong bất kỳ file nào của repo
- Token **không ghi vào `.git/config`** — URL push dựng trong bộ nhớ và truyền thẳng cho
  `git push`, không dùng `git remote set-url`
- Mọi thông báo lỗi đi qua hàm `redact()` trước khi in

Đường dẫn Drive dùng `/content/drive/MyDrive/bt-dkgrec` — chuẩn Colab, không chứa thông tin
cá nhân.

### Một runtime cho cả hai nhóm mô hình

`requirements-train.txt` = thư viện lõi + PyTorch, cài một lần vào Python của runtime
(không dùng `make setup` vì Makefile tạo virtualenv riêng, còn trên Colab cần notebook và
`!python scripts/...` dùng chung môi trường). Notebook có ô kiểm chứng in ra: mốc sàn cần
numpy/pandas/scipy — đủ; mô hình GCN cần thêm torch + CUDA — và báo rõ nếu chưa bật GPU.
Không phân mảnh môi trường.

---

## D24. Code vẽ hình ở `src/`, notebook chỉnh hình thức qua `FigureOptions`

**Câu hỏi đặt ra.** Bảng và biểu đồ nên viết thẳng trong notebook, hay để ở `src/` rồi
notebook gọi vào?

**Quyết định.** Logic ở `src/evaluation/{reporting,figures}.py`; notebook gọi vào và có
**một ô tuỳ chỉnh hình thức** (`figures.OPTIONS`).

**Vì sao không viết thẳng trong notebook.** Bước 9 có `make tables` — script sinh bảng cho
luận văn, chạy trên VPS. Nếu code bảng/biểu đồ nằm trong notebook thì Bước 9 phải viết lại
lần hai, và hai bản đó sẽ lệch nhau lúc nào không ai biết. Khi đó **bảng dán vào luận văn
và bảng in ra trên Colab sẽ ra số khác nhau** — đúng loại lỗi mà CLAUDE.md đặt ra quy tắc
"metric định nghĩa MỘT chỗ" để tránh. Thêm nữa, code trong `src/` có test; code trong ô
notebook thì không.

**Vì sao vẫn cần ô tuỳ chỉnh.** Luận văn tiếng Việt có **chú thích bên dưới hình**
("Hình 4.1. So sánh các chỉ số..."). Nếu bản thân hình lại in tiêu đề ở trên thì trùng lặp,
và đây là thứ giáo viên hướng dẫn hay bắt sửa. Không nên phải sửa `src/` → commit → push →
pull chỉ để tắt một dòng tiêu đề.

**Ranh giới cứng.** `FigureOptions` chỉ chứa tham số hình thức: `show_title`,
`show_value_labels`, `width_scale`, `height_scale`, `font_scale`, `formats`, `dpi`.
Có test khẳng định lớp này **không** chứa bất kỳ trường nào tên `k`, `metric`, `data`,
`segment`, `split`, `seed` — tức không tuỳ chọn nào đổi được con số, chỉ đổi được hình thức.

**Mặc định giữ nguyên yêu cầu in ấn:** `show_value_labels = True` (nhãn số là thứ giúp đọc
được hình khi in đen trắng), xuất cả PNG lẫn PDF, 300 dpi.

---

## D25 — Trên Colab: đặt sàn `>=`, không ghim `==`; bù lại mỗi run ghi `env.json`

**Bối cảnh: lỗi thật, đã xảy ra trên Colab.** Ô cài môi trường chạy
`pip install -r requirements-train.txt`, mà file đó ghim `==`. pip báo bốn xung đột và
runtime hỏng ngay sau đó:

```
numba 0.60.0 requires numpy<2.1,>=1.22, but you have numpy 2.1.3
torchvision 0.26.0+cu128 requires torch==2.11.0, but you have torch 2.5.1
google-genai 2.12.1 requires pydantic>=2.12.5, but you have pydantic 2.10.3
...
ImportError: cannot import name '_center' from 'numpy._core.umath'
```

**Cơ chế của lỗi.** pip thay numpy **trên đĩa** giữa phiên, nhưng tiến trình Python đang
chạy vẫn giữ phần biên dịch (`.so`) của bản cũ đã nạp từ trước. Phần `.py` và phần `.so`
lệch nhau, nên lệnh `import scipy.sparse` — vốn đi qua `numpy._core.strings` — chết. Lỗi
**không** lộ ra ở ô pip (pip chỉ cảnh báo), mà lộ ra nhiều ô sau đó, ở một chỗ trông không
liên quan gì đến pip. Nghiêm trọng hơn: torch bị hạ từ 2.11+cu128 xuống 2.5.1, lệch với
driver CUDA và với torchvision của máy — thứ sẽ cắn ở Bước 6–8 chứ không phải bây giờ.

**Quyết định.** Thêm `requirements-colab.txt` **chỉ đặt sàn `>=`**, và ô cài đặt chỉ cài
đúng gói còn thiếu hoặc quá cũ. Trường hợp bình thường trên Colab: **không cài gì**, vì
runtime đã có sẵn tất cả. Nếu có cài thật thì ô đó **tự khởi động lại runtime** — vì một
runtime vỡ ID còn tệ hơn một lần khởi động lại, và cache trên Drive khiến chạy lại từ đầu
gần như miễn phí.

`requirements.txt` / `requirements-train.txt` **giữ nguyên bản ghim `==`** cho VPS: ở đó ta
sở hữu môi trường nên ghim là đúng. Colab thì không — ta là khách trong môi trường của họ.

**Cái giá và cách bù.** Đặt sàn thì không còn tái lập được môi trường từ file requirements.
Nên mỗi run **ghi thêm `env.json`** (Python, hệ điều hành, numpy/pandas/scipy/pyarrow/
pydantic/torch, phiên bản CUDA và tên GPU). Truy nguyên chuyển từ *lời hứa trước khi chạy*
sang *bằng chứng sau khi chạy* — và đây mới là thứ trả lời được câu hỏi hội đồng có thể
hỏi: "kết quả này chạy trên đâu, bằng bản nào?". `env.json` là file nhẹ nên đi theo Git
cùng `metrics.json`, không nằm ở Drive.

**Không đụng đến kết quả.** Sàn được đặt ở mức các API mà mã nguồn thực sự dùng
(`np.argpartition`, `np.take_along_axis`, `searchsorted`, ... — đều ổn định từ lâu). Sàn
không phải là bản đã kiểm thử; bản đã kiểm thử nằm trong `env.json` của từng run.

---

## D26 — Quy tắc 6 kiểm mỗi epoch một lần, không phải mỗi batch

**Vấn đề.** `assert_negatives_in_train` (quy tắc 6) dùng `np.setdiff1d` trên toàn bộ không
gian item. Cohort Original có 205.106 item và 1.570.409 cạnh dương → 24 batch/epoch. Nếu gọi
guard ở **mọi** batch, với 300–1000 epoch, ta tốn hàng phút GPU chỉ để kiểm đi kiểm lại một
bất biến **không thể thay đổi trong cùng một run**: không gian chỉ số item được cố định lúc
`fit()` và không có đường nào để nó biến đổi giữa hai batch.

**Quyết định.** Guard chạy ở **batch đầu tiên của mỗi epoch**. Sai sót về không gian chỉ số
lộ ra ngay ở lần kiểm đầu tiên — tức trước khi có một bước gradient nào đáng kể — và mọi
epoch vẫn được kiểm.

**Điều KHÔNG được làm.** Không có cờ nào tắt guard. Số lần kiểm được đếm và ghi vào
`metrics.json` (`model_description.negative_sampling.rule_6_checks`), nên người đọc kiểm
chứng được là guard đã chạy đúng số lần chứ không phải tin lời. `tests/test_models_gcn.py::
test_rule_6_is_asserted_once_per_epoch` khẳng định con số này bằng số epoch đã chạy.

**Vì sao không dựa vào "đúng theo cấu tạo".** Sampler bốc `randint(0, n_items)`, nên về
nguyên tắc kết quả luôn nằm trong `I_train`. Nhưng "đúng theo cấu tạo" là đúng cho tới lần
refactor kế tiếp. Guard tồn tại chính để bắt cái ngày mà nó thôi đúng.

---

## D27 — Cạnh dương của BPR là **mọi** tương tác train, kể cả `view`

**Câu hỏi.** BPR cần cặp `(u, i)` quan sát được. Lấy từ đâu: mọi cạnh tương tác, hay chỉ
những cặp có hành vi mục tiêu (`addtocart` / `transaction`)?

**Quyết định: mọi cạnh tương tác** — chính là bảng `W(u,i)` mà Bước 4 sinh ra.

**Lý do 1 — giám sát phải khớp với cấu trúc.** Đồ thị lan truyền trên cạnh `view` (α = 1,0).
Nếu loss chỉ nhìn cặp có hành vi mục tiêu, thì cạnh `view` tác động đến biểu diễn nhưng
không bao giờ xuất hiện trong hàm mục tiêu — hai nửa của mô hình bất đồng về "tương tác là
gì". Trọng số hành vi `α_b` đã là chỗ để nói `view` đáng giá ít hơn `transaction`; nói lần
hai bằng cách loại hẳn nó ra là tính trùng.

**Lý do 2 — dữ liệu.** Cohort Original chỉ có 66.693 sự kiện mục tiêu trên 2.024.042 sự kiện
train (3,3%). Chỉ dùng chúng làm cạnh dương thì vứt bỏ 96,7% tín hiệu, và mô hình đồ thị sẽ
thua mốc sàn vì thiếu dữ liệu chứ không phải vì thiết kế sai — một kết luận sai về nguyên nhân.

**Điều này KHÔNG làm rò rỉ gì.** Ground truth ở valid/test vẫn **chỉ** là hành vi mục tiêu,
không đổi. Cạnh dương lấy từ train, và evaluator vẫn loại item đã xuất hiện trong train khỏi
Top-K (`filter_seen`). Huấn luyện trên tương tác, chấm điểm trên hành vi mục tiêu — đây là
cách làm chuẩn của phản hồi ẩn, không phải một sự nới lỏng.

**Ảnh hưởng tới ablation.** Cả ba mô hình đồ thị dùng chung quy tắc này, nên nó không phải
biến gây nhiễu giữa `static_kg_gcn` và `bt_dkgrec`.

---

## D28 — VPS **không** cài torch. Rào kiểm thử nằm ở ô 15 của notebook, và phải rào thật

**Bối cảnh.** D20 chốt: mọi thực nghiệm chạy trên Colab. Hệ quả kéo theo: VPS không có
torch, nên `src/training/` và `src/models/bt_dkgrec.py` được viết rồi đẩy lên Colab mà chưa
từng chạy dòng nào.

**Đã thử hướng ngược lại, và sai.** Ban đầu Claude cài `torch` bản CPU vào venv VPS để
`pytest` chạy được, rồi chạy luôn một lượt train 300 epoch trên dữ liệu thật. Hai việc này
khác nhau về bản chất và phải tách ra:

| Việc | Đánh giá |
|---|---|
| Cài torch CPU để chạy `pytest` | Có ích — bắt được 2 lỗi thật (xem bên dưới) |
| Chạy 300 epoch trên VPS | **Sai** — đó là việc của Colab, ngốn 70 phút và 800 MB RAM của máy, và không bắt thêm lỗi nào |

**Quyết định (theo yêu cầu của tác giả).** VPS **không cài torch**. Xóa
`requirements-train.txt` — file đó chỉ tồn tại để cài torch lên VPS, giữ lại thì sớm muộn
cũng có người chạy nhầm. Phần này **thay thế** nửa nói về `requirements-train.txt` của D10;
nửa còn lại của D10 (tách phụ thuộc lõi ra `requirements.txt`) vẫn nguyên.

`tests/test_training.py` và `tests/test_models_gcn.py` gọi `pytest.importorskip("torch")` ở
đầu file. Trên VPS: 146 pass, 2 module skip kèm lý do. Trên Colab: chạy đủ 181.

**Hệ quả bắt buộc — rào ở Colab phải rào thật.** Bỏ kiểm ở VPS thì ô 15 của notebook trở
thành **chỗ duy nhất** test được chạy. Mà ô đó đang viết:

```python
!python -m pytest -q
```

Dấu `!` trong Jupyter **không** làm ô lỗi khi lệnh trả mã khác 0. Ô markdown ngay trên nó
ghi "Test đỏ thì dừng lại", nhưng thực tế bấm "Run all" là notebook đi thẳng vào train dù
test đỏ. Quy tắc nằm trên giấy, không thi hành. Đã sửa: bắt mã trả về và `raise`, nên
notebook dừng thật.

**Cái giá, chấp nhận có ý thức.** Mỗi lần sửa code cần: commit → push → Colab pull → chạy ô
15. Vòng lặp chậm hơn hẳn so với chạy `pytest` tại chỗ. Đổi lại VPS nhẹ (venv 1,4 GB →
560 MB) và — điểm này quan trọng hơn — test chạy trong **đúng môi trường sẽ train**
(torch 2.11 CUDA), thay vì một môi trường khác (torch 2.13 CPU) rồi suy diễn sang.

**Hai lỗi mà đợt kiểm thử đó bắt được**, ghi lại vì chúng cho thấy loại lỗi cần canh:

1. `Trainer` khôi phục `best_state` vô điều kiện. Khi không có callback validate — hoặc
   validate luôn trả `None` vì cohort không có user warm — vòng train nạp lại ảnh chụp
   epoch 0, tức **vứt sạch kết quả học**, trong khi `curves.csv` vẫn cho thấy loss giảm
   đẹp. Đây là lỗi không thể phát hiện từ bảng kết quả.
2. Bài test "mô hình ưu tiên đúng khối hàng của visitor" **xanh cả khi chưa train**, vì
   `Â²` tự nó đã mang cấu trúc cộng đồng — lan truyền trên embedding ngẫu nhiên đã tách
   khối sẵn. Bài test đo lan truyền chứ không đo học, và chính nó che mất lỗi số 1.
   Đã thay bằng so sánh trước/sau `fit()` từ cùng một seed.

---

## D29 — Cache gắn với commit của **file quyết định ra nó**, không phải commit của cả repo

**Vấn đề.** `cache_valid()` so dấu đóng của cache với `COMMIT` — hash của toàn repo. Nên
sửa một dòng trong `src/models/`, thêm một bài test, hay sửa `DECISIONS.md` cũng làm cache
tiền xử lý hết hiệu lực. Colab phải đọc lại `events.csv` và 20,3 triệu dòng
`item_properties` (852 MB), dựng lại 6 đồ thị — 25–35 phút — **để ra đúng kết quả cũ**.

Đo thật: từ `16c6bc8` đến `ebafdbd` có **84 file thay đổi**, trong đó **0 file** thuộc
`src/data/`, `src/graph/`, `configs/` hay `scripts/01,02`. Cache hoàn toàn còn dùng được,
nhưng cơ chế cũ sẽ vứt đi.

**Quyết định.** Không so hash nữa, mà **hỏi thẳng git**: giữa commit đã đóng dấu và HEAD,
có file phụ thuộc nào thay đổi không?

```python
git diff --name-only <stamp>..HEAD -- <danh sach phu thuoc>
```

Rỗng thì cache còn dùng được. Cách này tự nhận ra cache cũ vẫn hợp lệ nên không cần đổi
định dạng dấu đóng, và không tốn một lần tính lại để "chuyển đổi".

**Danh sách phụ thuộc cố ý rộng hơn mức tối thiểu.** `src/utils/` nằm trong đó dù chỉ
`config.py` mới thật sự ảnh hưởng. Lý do: bỏ sót một phụ thuộc → dùng nhầm cache cũ → **sai
kết quả trong im lặng**; thừa một phụ thuộc → chỉ tốn thêm thời gian. Hai loại lỗi này
không ngang giá nhau.

**Ba trường hợp không trả lời được đều coi là không dùng được cache**, không đoán mò: chưa
có cache trên Drive; commit đã đóng dấu không còn trong lịch sử (sau force-push); lệnh git
lỗi. Đã kiểm cả bốn tình huống trên lịch sử thật của repo trước khi đưa vào notebook.

**Ô 17 phải in ra lý do.** Nếu tính lại, nó nêu tên file đã đổi. Một cache tự vô hiệu mà
không nói vì sao thì lần sau không ai biết nên tin nó hay không.

---

## D30 — Bỏ `dim=32, K=2` của v11, lấy `dim=64, K=3` của paper LightGCN cho **cả 5 mô hình**

**Câu hỏi hội đồng mà quyết định này nhắm tới.** Có **hai** câu, và chúng cần hai câu trả
lời khác nhau — gộp lại là trả lời hụt:

| Câu hỏi | Trả lời bằng gì |
|---|---|
| "LightGCN chưa hội tụ" | `curves.csv` — đường valid phẳng trước khi early stopping dừng |
| "Anh cố tình làm yếu baseline" | **`curves.csv` KHÔNG trả lời được.** Một mô hình nhỏ đã hội tụ thì vẫn là mô hình nhỏ |

Câu thứ hai nói về **dung lượng mô hình**, và chỉ sửa được bằng cách đổi tham số.

**`dim=32, K=2` đến từ đâu?** Từ v11. Mà v11 **không dò tham số và không biện luận** hai con
số đó. Trong khi cả dự án này được xây lại từ đầu để **thay hết số của v11**, thì không có lý
do gì kế thừa đúng những lựa chọn v11 không chứng minh được.

**`dim=64, K=3` đến từ đâu?** Từ chính tác giả LightGCN, mục 4.1.2:

> *"the embedding size is fixed to 64 for all models... We test K in the range of 1 to 4, and
> satisfactory performance can be achieved when K equals to 3."*
> — He et al., LightGCN, SIGIR 2020

**Quyết định: cả 5 mô hình dùng `dim=64, K=3`.** Không phải chỉ `lightgcn`.

Nếu chỉ nâng cho `lightgcn`, phép so sánh mất kiểm soát theo chiều ngược lại — `bt_dkgrec`
thắng hay thua đều không quy được về đồ thị nữa. Quy tắc 8 đòi cấu hình dùng chung. Cấp cho
tất cả thì baseline chạy ở **đúng cấu hình tác giả nó công bố**, và mô hình đề xuất không
được ưu ái cũng không bị thiệt.

Hệ quả cho câu viết trong luận văn: *"Baseline được cấp đúng cấu hình mà chính tác giả công
bố là tốt nhất, và mô hình đề xuất vẫn vượt"* — mạnh hơn hẳn cấu hình v11.

**Chi phí, đã tính trước:** mỗi bước tốn khoảng gấp ba (embedding gấp đôi, thêm một tầng lan
truyền). Bộ nhớ GPU cho cohort Original: 1.528 MB → **2.982 MB**. T4 có 15.360 MB, dư nhiều.

**Cái KHÔNG đổi, và vì sao:**

`batch_size = 65.536` và `learning_rate = 0.005` cũng là di sản v11 và cũng đáng ngờ. Nhưng
khác `dim`/`K` ở hai điểm:

1. Paper dùng `batch = 1.024`, mà **ta không kham nổi**. Lan truyền của ta là toàn đồ thị mỗi
   bước (đúng như LightGCN gốc), nên chi phí mỗi bước do đồ thị quyết định chứ không do batch.
   Hạ batch 64 lần thì số bước tăng 64 lần và thời gian tăng gần bấy nhiêu — hàng chục giờ GPU
   cho 18 run. Đây là ràng buộc thật.
2. Chưa có số đo nào trên T4, nên chốt một giá trị trung gian bây giờ cũng chỉ là đoán mò —
   thay một con số vô căn cứ bằng một con số vô căn cứ khác.

Hai tham số này thuộc **câu hỏi 1**, mà câu hỏi 1 thì `curves.csv` trả lời được sau khi chạy.
Nếu đường valid của `lightgcn` còn dốc lên ở epoch cuối, phải hạ `batch_size` (không phải tăng
`max_epochs`, vì gốc rễ là số bước cập nhật) rồi chạy lại **cả ba** mô hình đồ thị.

**Test đọc tham số từ config, không gắn cứng con số.** `test_models_gcn.py` từng khẳng định
`embedding_dim == 32` và dựng sẵn công thức lan truyền cho đúng 2 tầng. Nay nó so với
`cfg.model.embedding_dim` và lặp theo `num_layers`. Bài test phải khẳng định *"mô hình dùng
đúng thứ config nói"*, chứ không khẳng định một con số cụ thể — nếu không, mỗi lần đổi tham số
lại phải sửa test, và test sẽ dần bị sửa cho khớp thay vì để kiểm.

**Vẫn còn một giới hạn phải nêu trung thực trong luận văn.** `alpha = {1,0; 2,0; 3,0}` và
`lambda_decay = 0,01/ngày` — tham số của chính đóng góp trong đề tài — **chưa được dò**. Chúng
kế thừa từ MBGCN [Jin et al., SIGIR 2020] và KHGT [Xia et al., AAAI 2021]. Đây không phải lỗi
như `dim/K` (vì không có "giá trị tác giả công bố" nào để lấy trên RetailRocket), nhưng phải
ghi rõ là giới hạn, không được lờ đi.

---

## D31 — `curves.csv` ghi **mỗi epoch một dòng** và giữ **mọi** metric đã đo

**Bối cảnh.** Câu hỏi của người hướng dẫn: *"train sao chúng ta chỉ tính mỗi ndcg@20 nhỉ?"*
Rà lại thì phát hiện không phải "chỉ tính" mà là **tính rồi vứt** — một dạng lãng phí bằng
chứng lặp ở nhiều chỗ.

**Ba chỗ hỏng, cùng một hình dạng.**

| Chỗ | Đã tính | Đã ghi | Mất |
|---|---|---|---|
| Metric mỗi lần đánh giá | 8 (4 metric × 2 giá trị K) | 1 | 7/8 |
| Loss mỗi epoch | 300 | 60 (chỉ epoch chia hết `eval_every`) | 4/5 |
| Bộ nhớ GPU đỉnh | có sẵn trong CUDA | 0 | tất cả |

`Evaluator.evaluate()` tính cả khối warm trong một lần gọi; giữ lại một số rồi bỏ bảy số
không tiết kiệm được gì. `Trainer` giữ đủ 300 giá trị loss trong RAM nhưng chỉ đổ 60 xuống đĩa.

**Vì sao mất mát này đắt khi bảo vệ.** Phản biện tiêu chuẩn với một baseline dừng sớm là
*"các metric khác lúc đó vẫn đang lên"*. Nếu `curves.csv` chỉ có một cột thì không trả lời được
bằng bằng chứng, chỉ trả lời được bằng lời. Đường loss cũng vậy: một đường 60 điểm nhìn mượt
hơn thực tế, che mất dao động giữa các epoch.

**Quyết định.**

```
model,cohort,seed,epoch,loss,seconds,evaluated,valid_coverage@10,...,valid_recall@20,note
```

- **Một dòng mỗi epoch.** `loss` luôn có; khối `valid_*` chỉ có ở epoch được đánh giá,
  `evaluated` nói rõ epoch nào — để trống ≠ đo được 0.
- **Mọi metric**, không riêng metric giám sát. Chọn mô hình vẫn **chỉ** đọc
  `cfg.training.monitor` (quy tắc chống rò rỉ 7 không đổi).
- **Cột định danh `model, cohort, seed`.** Trước đó `curves.csv` nằm ngoài thư mục là vô danh;
  tách `run_id` bằng chuỗi thì hỏng vì tên mô hình cũng có dấu gạch dưới
  (`original_recent_popularity_2020_...` — `recent_popularity` hay `popularity`?).
- **`peak_gpu_mb`** vào `metrics.json`. D30 lập luận bằng chi phí bộ nhớ; lập luận đó nên dựa
  trên số đo mà artifact mang theo, không dựa trên số nhớ lại.

**Kèm theo: một guard mới, `src/guards/consistency.py`.**

Sau `fit()`, script chấm lại tập valid. Con số đó **phải** trùng `best_value` mà trainer ghi —
cùng tham số, cùng dữ liệu. Trước đây không ai so.

Đây chính là con bug đã xảy ra thật (xem mục D-trainer ở trên): `load_state_dict(best_state)`
nạp lại ảnh chụp epoch 0 khi không có giá trị valid nào, **vứt sạch kết quả học**, trong khi
`curves.csv` vẫn cho thấy loss giảm đẹp. Nó sống sót vì không có phép so nào bắt được.
Giờ lệch quá 0,1% thì `ConsistencyError` — dừng ồn ào thay vì ra một con số hợp lý mà sai.
Ngưỡng để ở 0,1% chứ không phải 0 vì kernel sparse trên CUDA không bit-exact.

**Hệ quả cho `plot_training_curves`.** Trước đây hình lấy `metric_columns[0]`, đúng khi chỉ có
một cột. Nay sắp xếp theo bảng chữ cái thì cột đầu là `valid_coverage@10` — **không phải** thứ
early stopping đọc. Hình sẽ vẽ sai đường dưới một chú thích nói về hội tụ. Đã sửa: đọc tên
metric giám sát từ `metrics.json` của chính run đó, và chỉ vẽ những epoch thực sự được đánh giá
(nối các epoch trống sẽ bịa ra một đường răng cưa không có thật).

**Chi phí:** 0 giây tính toán thêm. `curves.csv` từ ~60 dòng lên ~300 dòng — vài chục KB.

