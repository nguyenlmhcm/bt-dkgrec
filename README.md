# BT-DKGRec-GCN

Behavior-Time Dynamic Knowledge Graph + weighted GCN cho dự báo hành vi khách hàng
trên RetailRocket. Mã nguồn của đề án thạc sĩ *"Xây dựng ứng dụng dự báo hành vi
khách hàng sử dụng đồ thị tri thức động"*.

## Bắt đầu

```bash
make setup        # tạo .venv, cài thư viện lõi, kiểm tra môi trường + config
make test         # pytest
make help         # xem toàn bộ lệnh
```

Trên Colab (có GPU) chạy thêm `make setup-train` để cài PyTorch.

## Tài liệu

| File | Nội dung |
|---|---|
| `CLAUDE.md` | Đặc tả chính: bài toán, dữ liệu, siêu tham số, quy tắc bất di bất dịch |
| `docs/KG_DESIGN.md` | Thiết kế đồ thị tri thức, công thức, Cypher DDL |
| `docs/DECISIONS.md` | Nhật ký quyết định thiết kế + khác biệt so với bản v11 |
| `docs/De_an_thac_si_v11.docx` | Luận văn gốc, tham chiếu công thức (3.16)–(3.31) |

## Dữ liệu

`data/raw` là **symlink** tới `Datasets_RetailRocket_Raw/` (không commit, ~1,3 GB).
Tải lại từ [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
nếu mất.

## Tiến độ

- [x] **Bước 1** — Khung dự án, config loader, Makefile
- [x] **Bước 2** — Tầng dữ liệu (loader, splitter, mapping, side_info, cohort)
- [x] **Bước 3** — Leakage guards (7 rule + gate trong preprocess)
- [x] **Bước 4** — Tầng đồ thị (weighting, builder, normalize)
- [x] **Bước 5** — Interface + metrics + evaluator + popularity
- [ ] Bước 6 — bt_dkgrec + training
- [ ] Bước 7 — static_kg_gcn (subclass)
- [ ] Bước 8 — lightgcn (train tới hội tụ)
- [ ] Bước 9 — Multi-seed + bảng kết quả
- [ ] Bước 10 — Colab notebook
- [ ] Bước 11 — Export Neo4j + demo
