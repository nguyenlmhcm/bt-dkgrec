# Bảng phụ lục — tách loại hành vi (α) khỏi suy giảm thời gian (λ)

Sinh bởi `scripts/16_bang_ablation.py` từ `experiments/runs/` — không sửa tay.
Tập test, nhóm warm, 3 seed (2020, 2021, 2022). Tỉ lệ = (mô hình − Static KG-GCN)
/ (BT-DKGRec-GCN − Static KG-GCN), tính trên trung bình 3 seed.
Cột "thắng Static" đếm số seed mô hình cao hơn Static KG-GCN ở cùng seed.

## Original — Recall@20

| Cấu hình | α | λ | Seed 2020 | Seed 2021 | Seed 2022 | Trung bình | Tỉ lệ | Thắng Static |
|---|---|---|---:|---:|---:|---:|---:|:-:|
| Static KG-GCN (không có cả hai) | 1/1/1 | 0 | 0.029583 | 0.030597 | 0.021353 | 0.027178 | — | — |
| Chỉ loại hành vi | 1/2/3 | 0 | 0.028649 | 0.030949 | 0.022962 | 0.027520 | +10% | 2/3 |
| Chỉ thời gian | 1/1/1 | 0.05 | 0.024427 | 0.029922 | 0.024305 | 0.026218 | -29% | 1/3 |
| BT-DKGRec-GCN (cả hai) | 1/2/3 | 0.05 | 0.032813 | 0.029400 | 0.029146 | 0.030453 | +100% | 2/3 |

## Original — NDCG@20

| Cấu hình | α | λ | Seed 2020 | Seed 2021 | Seed 2022 | Trung bình | Tỉ lệ | Thắng Static |
|---|---|---|---:|---:|---:|---:|---:|:-:|
| Static KG-GCN (không có cả hai) | 1/1/1 | 0 | 0.018809 | 0.016349 | 0.012568 | 0.015909 | — | — |
| Chỉ loại hành vi | 1/2/3 | 0 | 0.019138 | 0.016342 | 0.013640 | 0.016373 | +26% | 2/3 |
| Chỉ thời gian | 1/1/1 | 0.05 | 0.014271 | 0.015249 | 0.017474 | 0.015665 | -14% | 1/3 |
| BT-DKGRec-GCN (cả hai) | 1/2/3 | 0.05 | 0.018531 | 0.015472 | 0.019108 | 0.017704 | +100% | 1/3 |

**Kết luận theo ngưỡng D35 (Original, Recall@20): hai thành phần bổ sung nhau.**

## Active — Recall@20

| Cấu hình | α | λ | Seed 2020 | Seed 2021 | Seed 2022 | Trung bình | Tỉ lệ | Thắng Static |
|---|---|---|---:|---:|---:|---:|---:|:-:|
| Static KG-GCN (không có cả hai) | 1/1/1 | 0 | 0.027101 | 0.031945 | 0.034906 | 0.031317 | — | — |
| Chỉ loại hành vi | 1/2/3 | 0 | 0.029344 | 0.034767 | 0.035974 | 0.033362 | +31% | 3/3 |
| Chỉ thời gian | 1/1/1 | 0.05 | 0.038835 | 0.042079 | 0.034970 | 0.038628 | +109% | 3/3 |
| BT-DKGRec-GCN (cả hai) | 1/2/3 | 0.05 | 0.044177 | 0.035932 | 0.033901 | 0.038004 | +100% | 2/3 |

## Active — NDCG@20

| Cấu hình | α | λ | Seed 2020 | Seed 2021 | Seed 2022 | Trung bình | Tỉ lệ | Thắng Static |
|---|---|---|---:|---:|---:|---:|---:|:-:|
| Static KG-GCN (không có cả hai) | 1/1/1 | 0 | 0.020711 | 0.026246 | 0.023759 | 0.023572 | — | — |
| Chỉ loại hành vi | 1/2/3 | 0 | 0.021012 | 0.027105 | 0.024309 | 0.024142 | +24% | 3/3 |
| Chỉ thời gian | 1/1/1 | 0.05 | 0.025368 | 0.028908 | 0.022733 | 0.025670 | +89% | 2/3 |
| BT-DKGRec-GCN (cả hai) | 1/2/3 | 0.05 | 0.026007 | 0.028586 | 0.023155 | 0.025916 | +100% | 2/3 |

**Kết luận theo ngưỡng D35 (Active, Recall@20): chủ yếu từ thời gian.**

