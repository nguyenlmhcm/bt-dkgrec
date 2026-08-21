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
