# BT-DKGRec-GCN

Behavior-Time Dynamic Knowledge Graph cho dự báo hành vi khách hàng trên RetailRocket.

Luận văn thạc sĩ — "Xây dựng ứng dụng dự báo hành vi khách hàng sử dụng đồ thị tri thức động".

---

## Bối cảnh quan trọng

Codebase gốc **đã mất**. Dự án này xây lại **hoàn toàn từ đầu**, dùng luận văn v11 làm đặc tả duy nhất.

Hệ quả cần nắm:

| | |
|---|---|
| Tài sản còn lại | Luận văn v11 (.docx) + dữ liệu RetailRocket raw |
| Không thể làm | Tái lập chính xác số liệu cũ — không còn code tham chiếu |
| Phải làm | Chạy lại toàn bộ, **thay hết số trong luận văn** bằng kết quả mới |
| Tuyệt đối tránh | Tinh chỉnh code cho ra khớp số cũ — đó là làm ngược quy trình khoa học |

Các thống kê trong tài liệu này dùng làm **mốc kiểm tra hợp lý** (sanity check), không phải mục tiêu phải khớp. Lệch ±5% ở số node/cạnh là bình thường; lệch một bậc độ lớn nghĩa là có bug.

---

## Bài toán

| | |
|---|---|
| Input | Chuỗi sự kiện `(visitor, item, behavior, timestamp)` |
| Behavior | `view`, `addtocart`, `transaction` |
| Target | `addtocart` **OR** `transaction`; `view` chỉ là tín hiệu lịch sử |
| Task | Top-K item có khả năng phát sinh hành vi mục tiêu |
| Metric | Recall@{10,20}, NDCG@{10,20}, HitRate@{10,20}, Coverage@{10,20} — bảng chính K=20 |
| Split | Thời gian 70/10/20 — **không random split** |

Bài toán **ranking**, không phải classification. Không dùng Accuracy.

---

## Đặc tả dữ liệu RetailRocket

### Audit kỳ vọng trên raw

| Chỉ số | Giá trị kỳ vọng |
|---|---|
| `events.csv` tổng dòng | 2.756.101 |
| — view | 2.664.312 |
| — addtocart | 69.332 |
| — transaction | 22.457 |
| Target events (cart OR txn) | 91.789 |
| Item có property/category | 185.246 / 235.061 (78,81%) |
| Category nối được vào tree | 1.212 / 1.242 (97,58%) |

Script `01_preprocess.py` phải in bảng audit này. Nếu lệch nhiều → sai file raw hoặc sai cách đọc.

### Hai nhóm người dùng

Đề tài đánh giá trên hai cohort. Cả hai đều phải chạy.

| | Original | Active (`|E_u^train| ≥ 5`) |
|---|---|---|
| Train events | 2.024.042 | 707.680 |
| Train visitors | 1.027.985 | 60.559 |
| Train items | 205.106 | 86.659 |
| Train target events | 66.693 | 48.482 |
| Valid warm users | 552 | 234 |
| **Test warm users** | **593** | **234** |
| Cạnh tương tác sau tổng hợp | 1.570.409 | 405.128 |
| Side edges | 3.473.983 | 1.784.845 |
| — item-category | 165.528 | 81.135 |
| — item-property | 3.307.294 | 1.702.617 |
| — category-parent | 1.161 | 1.093 |
| Tổng entity trong graph | 214.396 | 131.223 |

**Lưu ý về ngưỡng ≥5:** ngưỡng này được chốt trước khi dựng lại mapping/graph. Kết quả trên cohort Active là **phân tích phân nhóm thăm dò**, không thay thế tập đánh giá chính (Original).

### Warm vs Cold user

- **Warm user**: có lịch sử trong train VÀ có target trong giai đoạn đánh giá
- **Cold user**: không có embedding cá nhân hóa từ train → báo cáo **riêng**, hoặc dùng phương án dự phòng theo độ phổ biến
- **Không trộn** warm và cold vào cùng metric của mô hình cá nhân hóa
- `warm` được báo cáo thêm theo **bậc** (`warm_deg1/2/3plus` — số cạnh của user trong
  train). Chuẩn hóa đối xứng đưa trọng số vào theo **căn bậc hai** — với user bậc 1 hệ
  số là `√W/√dᵢ` — nên tỉ lệ 3:1 giữa `transaction` và `view` còn `√3 ≈ 1,73:1`. Cơ chế
  tác động **nhẹ hơn danh nghĩa nhưng trên mọi dải bậc**. Tách theo bậc để thấy hiệu quả
  đến từ nhóm nào. Xem D34, gồm cả phần đính chính lập luận "triệt tiêu" đã bị bác bỏ.

### Chính sách candidate bảo thủ

Target thuộc item chưa xuất hiện trong train **không thể** xếp hạng → tính là **miss**, không loại khỏi mẫu số. Cách tính này nhất quán với `candidate = I_train` và tạo đánh giá thiên về bảo thủ.

---

## Kiến trúc mô hình

```
events (train) ──► Behavior-Time DKG ──► weighted GCN ──► BPR ──► Top-K
                   w = α_b·exp(-λΔt)     Â = D^-½AD^-½
                   W(u,i) = Σ w          z = mean(h^0..h^L)
                                         s(u,i) = z_u · z_i
```

Chi tiết đồ thị, công thức, Cypher DDL: **`docs/KG_DESIGN.md`**.

Công thức tham chiếu số hiệu trong luận văn: (3.16)–(3.18) trọng số, (3.24)–(3.27) lan truyền, (3.29)–(3.31) loss.

---

## Ma trận thực nghiệm

| Nhóm | Mô hình | Vai trò |
|---|---|---|
| Không cá nhân hóa | `popularity` | Mốc sàn |
| Có yếu tố thời gian | `recent_popularity` | Mốc sàn có tính gần đây |
| Graph CF | `lightgcn` | Baseline học thuật — không dùng side info, không behavior-time |
| KG tĩnh | `static_kg_gcn` | Ablation — có side info, **không** behavior-time |
| Đề xuất | `bt_dkgrec` | Đầy đủ — `λ = 0,01` kế thừa từ MBGCN/KHGT, **chưa dò** |
| Đề xuất (dò λ) | `bt_dkgrec_l05` | Đầy đủ — `λ = 0,05` dò trên tập xác thực (D33) |

**Lập luận cốt lõi của luận văn** nằm ở cặp `static_kg_gcn` vs `bt_dkgrec`. Hai mô hình phải khác **đúng một biến**: hàm `edge_weight()`. Mọi thứ khác dùng chung code.

`bt_dkgrec_l05` khác `bt_dkgrec` **đúng một tham số**: `weighting.lambda_decay`. Giữ cả hai trong bảng để người đọc thấy được việc dò tham số, chứ không chỉ thấy kết quả sau khi dò.

Lưu ý: `static_kg_gcn` **không phải** bản tái lập KGAT hay KGCN. Phải ghi rõ điều này khi viết luận văn để tránh hiểu nhầm.

---

## Siêu tham số

### Cấu hình cũ (v11) — dùng làm điểm khởi đầu

```yaml
embedding_dim: 32
num_layers: 2
batch_size: 65536
learning_rate: 0.005
reg_weight: 0.0001
epochs: 10          # ⚠️ XEM CẢNH BÁO BÊN DƯỚI
seed: 2026          # ⚠️ chỉ 1 seed
device: cuda
```

### Trọng số behavior-time

```yaml
alpha:
  view: 1.0
  addtocart: 2.0
  transaction: 3.0
lambda_decay: 0.01
d_day: 86400000     # ms/ngày — timestamp RetailRocket dùng ms
```

### Giới hạn graph

```yaml
max_property_edges: 50000   # giới hạn số NODE property=value phổ biến
                            # KHÔNG giới hạn tổng cạnh item-property
min_active_events: 5        # ngưỡng cohort Active
```

### ⚠️ Hai sai sót của cấu hình cũ — phải sửa

| Sai sót | Vì sao là vấn đề | Cách sửa |
|---|---|---|
| `epochs: 10` cho **mọi** mô hình | LightGCN gốc train 1000 epoch → baseline chưa hội tụ → so sánh không hợp lệ | Thêm early stopping theo valid, `max_epochs` đủ lớn (≥ 300), lưu `curves.csv` chứng minh hội tụ |
| Một seed duy nhất (2026) | Không phân biệt được cải thiện thật với nhiễu | Chạy 3 seed `[2020, 2021, 2022]`, báo cáo mean ± std |

Đây là hai lỗ hổng nặng nhất của bản v11. Code mới **phải** khắc phục cả hai.

---

## Quy tắc bất di bất dịch

Vi phạm bất kỳ điều nào sẽ làm kết quả mất giá trị học thuật.

### Chống rò rỉ dữ liệu

1. Không random split — luôn split theo thời gian
2. ID mapping **chỉ** tạo từ train
3. Side information chỉ nhận bản ghi `timestamp ≤ T_train`
4. Với item có nhiều bản ghi category theo thời gian → lấy **bản ghi mới nhất ≤ T_train**
5. Candidate set ⊆ `I_train`
6. Negative sampling chỉ lấy từ `I_train`
7. Model selection chỉ đọc metric của **valid** — không bao giờ theo test

Tất cả đã cài thành assertion trong `src/guards/leakage.py`. **Không được tắt guard để pipeline chạy qua.** Guard fail = bug thật.

Quy tắc 4 là bẫy dễ sót nhất: `item_properties` có nhiều dòng cùng item ở các timestamp khác nhau. Lấy dòng cuối file = nhìn thấy tương lai.

### Công bằng khi so sánh

8. Mọi mô hình dùng **chung** split, mapping, candidate set, evaluator
9. Baseline phải train **tới hội tụ** — không dừng sớm
10. Mọi mô hình chạy **cùng bộ seed**
11. Script sinh bảng đọc **toàn bộ** run trong `experiments/runs/` — không có tham số lọc seed

### Liêm chính

12. Không chọn seed cho kết quả đẹp
13. Không lấy số từ paper khác đưa vào bảng như thể tự chạy
14. Mô hình đề xuất thua baseline → **báo cáo trung thực**, phân tích nguyên nhân

---

## Cấu trúc repo

```
configs/          YAML — mọi siêu tham số, không hardcode
data/raw/         events.csv, item_properties_part{1,2}.csv, category_tree.csv
data/interim/     split + mapping
data/processed/   graph + trọng số cạnh
src/data/         loader, splitter, mapping, side_info, cohort
src/graph/        schema, weighting, builder, normalize
src/models/       base (interface), bt_dkgrec, static_kg_gcn, lightgcn, popularity
src/training/     loss (BPR), sampler, trainer, seeding
src/evaluation/   metrics, evaluator, segments, stats
src/guards/       leakage  ← không được bỏ qua
src/export/       neo4j_export
scripts/          01_preprocess → 07_export_neo4j
app/              demo Streamlit + Neo4j
experiments/      seeds.json, runs/<run_id>/
tests/            pytest
docs/             KG_DESIGN.md, PLAN.md
```

---

## Lệnh

```bash
make setup
make preprocess COHORT=original      # hoặc COHORT=active
make graph COHORT=original
make train MODEL=bt_dkgrec SEED=2020 COHORT=original
make evaluate RUN=<run_id>
make multiseed                       # toàn bộ model × seed × cohort
make tables                          # sinh bảng mean±std cho luận văn
make neo4j
make app
make test
```

---

## Chuẩn code

| Hạng mục | Quy ước |
|---|---|
| Python | 3.10+, type hints ở public API |
| Framework | PyTorch, scipy.sparse cho graph |
| Config | YAML + pydantic validate |
| Dữ liệu trung gian | Parquet |
| Log | `logging` → `experiments/runs/<run_id>/train.log` |
| Docstring & comment | Tiếng Anh |
| Tài liệu `docs/` | Tiếng Việt |

### DRY — ba quy tắc cứng

- `static_kg_gcn` **kế thừa** `bt_dkgrec`, chỉ override `edge_weight()`. Không copy file.
- Công thức trọng số định nghĩa **một chỗ**: `src/graph/weighting.py`
- Metric định nghĩa **một chỗ**: `src/evaluation/metrics.py`

### Non-determinism

Sparse ops trên CUDA không bit-exact dù cố định seed:

- Không viết test kỳ vọng bit-exact trên GPU
- Báo cáo `mean ± std` trên 3 seed
- So sánh dùng **Welch's t-test** (hai mẫu độc lập), không dùng paired test

---

## Run artifact

```
experiments/runs/<cohort>_<model>_<seed>_<timestamp>/
├── config.yaml      # snapshot config đã dùng
├── seed.txt
├── metrics.json     # Recall/NDCG/HitRate/Coverage @10,@20
│                   # warm / warm_deg1 / warm_deg2 / warm_deg3plus / cold / all
├── topk.csv         # Top-K từng visitor — demo đọc file này
├── curves.csv       # MỘT DÒNG MỖI EPOCH ← chứng minh hội tụ
├── env.json         # phiên bản thư viện + GPU thực tế đã chạy (D25)
└── train.log
```

`curves.csv` bắt buộc, schema chốt ở D31:

```
model,cohort,seed,epoch,loss,seconds,evaluated,valid_<metric>@<K>...,note
```

- `loss` có ở **mọi** epoch; khối `valid_*` chỉ có ở epoch được đánh giá (`evaluated`)
- giữ **mọi** metric đã đo, không riêng metric giám sát — evaluator tính cả khối trong một lần gọi
- chọn mô hình vẫn chỉ đọc `training.monitor` (quy tắc chống rò rỉ 7)

Khi bảo vệ, cần chứng minh LightGCN đã hội tụ chứ không bị dừng sớm — đây là câu hỏi hội đồng nhiều khả năng sẽ hỏi. Câu hỏi tiếp theo thường là *"các metric khác lúc đó thế nào?"*, nên một cột là không đủ.

---

## Thứ tự triển khai

Làm tuần tự, mỗi bước phải xanh trước khi sang bước sau.

| Bước | Việc | Điều kiện hoàn thành |
|:-:|---|---|
| 1 | Khung thư mục + config loader + Makefile | `make setup` chạy được |
| 2 | `src/data/` — loader, splitter, mapping, side_info, cohort | Bảng audit khớp mục "Đặc tả dữ liệu" |
| 3 | `src/guards/leakage.py` + tests | Toàn bộ guard pass |
| 4 | `src/graph/` — weighting, builder, normalize | `graph_stats.json` hợp lý so với Bảng thống kê |
| 5 | `src/models/base.py` + `popularity` | Pipeline end-to-end chạy được với mô hình đơn giản nhất |
| 6 | `bt_dkgrec` + `src/training/` | Train chạy, loss giảm, `curves.csv` sinh ra |
| 7 | `static_kg_gcn` như subclass | `git diff` chỉ khác ở `edge_weight()` |
| 8 | `lightgcn` | Hội tụ — `curves.csv` cho thấy plateau |
| 9 | `05_run_multiseed.py` + `06_make_tables.py` | Bảng mean±std tự sinh |
| 10 | `src/export/` + `app/` | Demo truy vết được một visitor thật |

**Bước 5 quan trọng hơn vẻ ngoài:** chạy `popularity` xuyên suốt pipeline trước khi làm mô hình phức tạp giúp phát hiện sớm lỗi ở evaluator, candidate set, hoặc metric — những chỗ nếu sai sẽ làm hỏng mọi kết quả sau đó.

---

## Demo — giới hạn có chủ đích

App **không tự tính điểm**. Nó đọc `topk.csv` từ run artifact (offline evaluation replay) và truy vấn Neo4j để hiển thị lịch sử, trọng số cạnh, subgraph.

Giới hạn này đã được nêu trung thực trong luận văn (mục 3.7.2) — **giữ nguyên**. Không để app tự suy luận rồi sinh số khác với bảng kết quả.

---

## Tài liệu nội bộ

- `docs/KG_DESIGN.md` — thiết kế đồ thị, công thức, Cypher DDL
- `docs/PLAN.md` — kế hoạch hoàn thiện luận văn
- Luận văn v11 (.docx) — đặc tả gốc, tham chiếu công thức (3.16)–(3.31)
