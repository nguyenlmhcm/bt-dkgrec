# Thiết kế Behavior-Time Dynamic Knowledge Graph

**Dataset:** RetailRocket | **Store:** Neo4j (lớp truy vết) + scipy.sparse (lớp huấn luyện)

---

## 1. Kiến trúc hai lớp

Đây là quyết định thiết kế cốt lõi, kế thừa từ mục 3.3 đề án.

| | Lớp truy vết (Trace Layer) | Lớp chiếu (Projected Layer) |
|---|---|---|
| Lưu ở | Neo4j | `scipy.sparse` → `torch.sparse` |
| Mục đích | Truy vấn, giải thích, trực quan hóa, demo | Học biểu diễn (GCN propagation) |
| Event | Vật thể hóa thành node `:Event` | Chiếu thành trọng số cạnh |
| Node types | Visitor, Item, Category, PropertyValue, Event | Visitor, Item, Category, PropertyValue |
| Cạnh visitor-item | Nhiều cạnh (mỗi sự kiện một cạnh) | **Một** cạnh, trọng số tổng hợp `W(u,i)` |

**Lý do tách:** giữ Event làm node trong graph huấn luyện sẽ làm đồ thị phình rất lớn (RetailRocket có ~2.7M events) mà không thêm tín hiệu học. Ngược lại, demo cần Event để truy vết từng hành vi.

**Ràng buộc nhất quán:** cả hai lớp phải sinh từ **cùng một** tập train và cùng `T_train`. Guard `assert_layers_consistent()` kiểm tra tổng trọng số cạnh của hai lớp khớp nhau.

---

## 2. Node types

### 2.1. `:Visitor`

| Property | Kiểu | Nguồn | Ghi chú |
|---|---|---|---|
| `visitor_id` | int | `events.visitorid` | Định danh gốc, **UNIQUE** |
| `idx` | int | mapping | Chỉ số ma trận, **chỉ tạo từ train** |
| `first_seen` | long | min(timestamp) | ms |
| `last_seen` | long | max(timestamp) | ms |
| `n_events` | int | count | Phục vụ phân nhóm warm/cold |
| `segment` | string | derived | `warm` \| `cold` |

### 2.2. `:Item`

| Property | Kiểu | Nguồn |
|---|---|---|
| `item_id` | int | `events.itemid` — **UNIQUE** |
| `idx` | int | mapping (train-only) |
| `first_seen` / `last_seen` | long | events |
| `n_interactions` | int | count |

### 2.3. `:Category`

| Property | Kiểu | Nguồn |
|---|---|---|
| `category_id` | int | `category_tree.categoryid` — **UNIQUE** |
| `idx` | int | mapping |
| `depth` | int | tính từ cây (root = 0) |
| `is_root` | bool | `parentid IS NULL` |

### 2.4. `:PropertyValue`

RetailRocket có `property` và `value` ẩn danh (đã hash). Gộp thành một node composite.

| Property | Kiểu | Nguồn |
|---|---|---|
| `prop_key` | string | `item_properties.property` |
| `prop_value` | string | `item_properties.value` |
| `pv_id` | string | `f"{prop_key}::{prop_value}"` — **UNIQUE** |
| `idx` | int | mapping |

**Lọc bắt buộc:**
- Bỏ dòng `property == 'categoryid'` (dùng để tạo `HAS_CATEGORY`, không tạo PropertyValue)
- Bỏ dòng `property == 'available'` (tín hiệu tồn kho, không phải tri thức sản phẩm)
- Bỏ PropertyValue có tần suất < `min_pv_freq` (mặc định 5) — tránh nổ số node

### 2.5. `:Event` — **chỉ ở lớp truy vết**

| Property | Kiểu | Nguồn |
|---|---|---|
| `event_id` | string | `f"{visitorid}_{itemid}_{timestamp}"` |
| `behavior` | string | `view` \| `addtocart` \| `transaction` |
| `timestamp` | long | ms |
| `w_event` | float | `α_b · exp(-λ·Δt)` — trọng số sự kiện |

---

## 3. Relation types

| Relation | Từ → Đến | Lớp | Properties |
|---|---|:---:|---|
| `INTERACTED_WITH` | Visitor → Item | Cả hai | `weight`, `n_view`, `n_cart`, `n_txn`, `last_ts` |
| `PERFORMED` | Visitor → Event | Truy vết | — |
| `TARGETS` | Event → Item | Truy vết | — |
| `HAS_CATEGORY` | Item → Category | Cả hai | `valid_from`, `rel_weight` |
| `HAS_PROPERTY` | Item → PropertyValue | Cả hai | `valid_from`, `rel_weight` |
| `PARENT_CATEGORY` | Category → Category | Cả hai | `rel_weight` |

`rel_weight` mặc định `1.0` cho mọi cạnh side information (theo cấu hình báo cáo của v11).

---

## 4. Cơ chế trọng số hành vi-thời gian

Đây là **đóng góp chính** của đề tài — cần cài đặt chính xác.

### 4.1. Công thức

```
Δt(u,i,b,t) = max(0, (T_train − t) / 86_400_000)        # tuổi sự kiện, đơn vị ngày
w(u,i,b,t)  = α_b · exp(−λ · Δt(u,i,b,t))               # trọng số sự kiện
W(u,i)      = Σ_{(b,t) ∈ E_train(u,i)} w(u,i,b,t)       # trọng số cạnh tổng hợp
```

### 4.2. Tham số (cấu hình báo cáo)

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `α_view` | 1.0 | Tín hiệu quan tâm yếu |
| `α_addtocart` | 2.0 | Ý định trung bình/cao |
| `α_transaction` | 3.0 | Chuyển đổi mạnh |
| `λ` | 0.01 | Hệ số suy giảm theo ngày |
| `D_day` | 86400000 | ms/ngày (timestamp RetailRocket dùng ms) |

Tất cả nằm trong `configs/models/bt_dkgrec.yaml` — **không hardcode**.

### 4.3. Ablation: Static KG-GCN

Biến thể tĩnh **chỉ khác đúng một chỗ**:

```python
# bt_dkgrec.py
def edge_weight(self, behavior, delta_t):
    return self.alpha[behavior] * math.exp(-self.lam * delta_t)

# static_kg_gcn.py  (kế thừa BTDKGRec)
def edge_weight(self, behavior, delta_t):
    return 1.0   # bỏ cả behavior lẫn time decay
```

Mọi thứ còn lại — graph builder, propagation, loss, evaluator — dùng chung. Đây là điều làm ablation trở thành so sánh có kiểm soát tuyệt đối.

---

## 5. Chống rò rỉ dữ liệu

Quy tắc bắt buộc khi dựng graph:

| # | Quy tắc | Guard |
|:-:|---|---|
| 1 | Node Visitor/Item **chỉ** tạo từ train | `assert_train_only_mapping()` |
| 2 | Cạnh `INTERACTED_WITH` chỉ từ events trong train | `assert_edges_in_train()` |
| 3 | `HAS_CATEGORY` / `HAS_PROPERTY` chỉ nhận bản ghi `timestamp ≤ T_train` | `assert_side_info_cutoff()` |
| 4 | Với item có nhiều bản ghi category theo thời gian → lấy bản ghi **mới nhất ≤ T_train** | `assert_single_category_per_item()` |
| 5 | Candidate set khi đánh giá ⊆ `I_train` | `assert_candidate_scope()` |

Quy tắc 4 dễ bị bỏ sót: `item_properties` của RetailRocket có nhiều dòng cùng item với timestamp khác nhau (thuộc tính thay đổi theo thời gian). Phải chọn snapshot tại `T_train`, không lấy dòng cuối cùng của file.

---

## 6. Chiếu sang ma trận huấn luyện

### 6.1. Không gian node hợp nhất

Gộp 4 loại node vào một chỉ số liên tục:

```
[0, n_visitor)                          → Visitor
[n_visitor, n_visitor+n_item)           → Item
[+n_item, +n_category)                  → Category
[+n_category, +n_pv)                    → PropertyValue
```

Lưu offset vào `mappings.json` để `evaluator` cắt đúng khối Visitor × Item khi tính điểm.

### 6.2. Ma trận kề & chuẩn hóa

```
A  = ma trận kề đối xứng (thưa), giá trị = ω(v,j)
d_v = Σ_{j ∈ N(v)} ω(v,j)                      # bậc có trọng số
Â   = D^(-1/2) · A · D^(-1/2)                  # chuẩn hóa đối xứng
```

### 6.3. Lan truyền

```
h^(l+1) = Â · h^(l)                            # không tham số, phong cách LightGCN
z       = (1/(L+1)) · Σ_{l=0..L} h^(l)         # mean pooling qua các lớp
s(u,i)  = z_u · z_i                            # tích vô hướng
```

**Chú ý:** chỉ embedding **lớp 0** là tham số học được. Các lớp sau không có ma trận trọng số — giữ đúng tinh thần LightGCN và làm nổi bật đóng góp nằm ở *cấu trúc đồ thị*, không phải ở độ phức tạp mạng.

---

## 7. Cypher DDL

### 7.1. Constraints & indexes

```cypher
CREATE CONSTRAINT visitor_id IF NOT EXISTS
  FOR (v:Visitor) REQUIRE v.visitor_id IS UNIQUE;

CREATE CONSTRAINT item_id IF NOT EXISTS
  FOR (i:Item) REQUIRE i.item_id IS UNIQUE;

CREATE CONSTRAINT category_id IF NOT EXISTS
  FOR (c:Category) REQUIRE c.category_id IS UNIQUE;

CREATE CONSTRAINT pv_id IF NOT EXISTS
  FOR (p:PropertyValue) REQUIRE p.pv_id IS UNIQUE;

CREATE INDEX visitor_segment IF NOT EXISTS FOR (v:Visitor) ON (v.segment);
CREATE INDEX item_idx        IF NOT EXISTS FOR (i:Item)    ON (i.idx);
```

### 7.2. Nạp dữ liệu (bulk, dùng `LOAD CSV`)

```cypher
// Visitor
LOAD CSV WITH HEADERS FROM 'file:///visitors.csv' AS row
CALL { WITH row
  CREATE (:Visitor {
    visitor_id: toInteger(row.visitor_id),
    idx:        toInteger(row.idx),
    first_seen: toInteger(row.first_seen),
    last_seen:  toInteger(row.last_seen),
    n_events:   toInteger(row.n_events),
    segment:    row.segment })
} IN TRANSACTIONS OF 10000 ROWS;

// Cạnh tương tác đã tổng hợp
LOAD CSV WITH HEADERS FROM 'file:///edges_interacted.csv' AS row
CALL { WITH row
  MATCH (v:Visitor {visitor_id: toInteger(row.visitor_id)})
  MATCH (i:Item    {item_id:    toInteger(row.item_id)})
  CREATE (v)-[:INTERACTED_WITH {
    weight:  toFloat(row.weight),
    n_view:  toInteger(row.n_view),
    n_cart:  toInteger(row.n_cart),
    n_txn:   toInteger(row.n_txn),
    last_ts: toInteger(row.last_ts) }]->(i)
} IN TRANSACTIONS OF 10000 ROWS;
```

### 7.3. Truy vấn cho demo

```cypher
// Lịch sử tương tác của một visitor, kèm trọng số
MATCH (v:Visitor {visitor_id: $vid})-[r:INTERACTED_WITH]->(i:Item)
RETURN i.item_id, r.weight, r.n_view, r.n_cart, r.n_txn, r.last_ts
ORDER BY r.weight DESC LIMIT 20;

// Subgraph Visitor → Item → Category/Property (phục vụ trực quan hóa)
MATCH (v:Visitor {visitor_id: $vid})-[r:INTERACTED_WITH]->(i:Item)
OPTIONAL MATCH (i)-[:HAS_CATEGORY]->(c:Category)
OPTIONAL MATCH (i)-[:HAS_PROPERTY]->(p:PropertyValue)
RETURN v, r, i, c, p LIMIT 100;

// Truy vết từng sự kiện (lớp truy vết)
MATCH (v:Visitor {visitor_id: $vid})-[:PERFORMED]->(e:Event)-[:TARGETS]->(i:Item)
RETURN e.behavior, e.timestamp, e.w_event, i.item_id
ORDER BY e.timestamp DESC LIMIT 50;
```

---

## 8. Kiểm chứng graph sau khi dựng

Chạy tự động sau `02_build_graph.py`, ghi vào `data/processed/graph_stats.json`:

| Chỉ số | Kỳ vọng |
|---|---|
| `n_visitor`, `n_item`, `n_category`, `n_pv` | Khớp mapping |
| `n_edge_interacted` | = số cặp (u,i) duy nhất trong train |
| `n_edge_category` | ≤ `n_item` (mỗi item ≤ 1 category tại `T_train`) |
| `weight_min` | > 0 (không có cạnh trọng số 0) |
| `weight_max` | ≤ `α_transaction × n_events_max` |
| `isolated_nodes` | = 0 sau khi lọc |
| `max(timestamp)` mọi cạnh | ≤ `T_train` |

Nếu bất kỳ dòng nào lệch → dừng, không train.

---

*Tài liệu lập ngày 20/8/2026.*
