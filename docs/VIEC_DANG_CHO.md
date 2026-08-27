# Việc đang chờ

Cập nhật: 2026-08-26, 14:45 (+07)

## Đang chạy

**12 run trainable của cohort Original** trên Colab, nhánh `exp/active-patience50`.

| | |
|---|---|
| Đã xong | 7/18 (6 heuristic + `lightgcn/2020`) |
| Tốc độ | ~33 phút/run |
| Còn lại | 11 run ≈ **7,5 giờ** |
| GPU | T4, ~1.07 đơn vị/giờ, cần ~12 trên tổng 40.73 |

Lưu ý: timestamp trong tên run là **UTC**, không phải giờ VN.

### Khi chạy xong

```bash
git pull
.venv/bin/python scripts/08_make_docx.py     # bảng tự cập nhật từ experiments/runs/
make tables                                   # bảng Markdown
.venv/bin/python scripts/analysis/degree_bands.py   # phân tầng theo bậc
```

**Con số cần nhìn:** cả ba dải bậc trên cohort Original. Phân bố thật trong 593
user được đánh giá: bậc 1 là 259 (43,7%), bậc 2 là 76 (12,8%), bậc ≥3 là 258
(43,5%). Con số 79,6% từng ghi ở đây là của 1.027.985 visitor trong tập train,
không phải của tập được đánh giá.

**Tiên đoán cũ đã bị bác bỏ.** Trước đây mục này ghi `warm_deg1` phải cho hiệu số
đúng bằng 0. Sai: chuẩn hoá đối xứng cho hệ số `√W/√dᵢ`, tức trọng số vào theo
căn bậc hai chứ không triệt tiêu. Đo trên Original seed 2020 cho `warm_deg1`
lệch +0.00386, phù hợp với công thức đúng. Việc triệt tiêu chỉ xảy ra với chuẩn
hoá theo hàng `D⁻¹A`, không phải dạng đối xứng mà mô hình dùng.

**Kiểm tái lập:** số liệu Original mới phải trùng bản trên `main`. Lệch thì phải
điều tra trước khi dùng.

## Trạng thái hai nhánh

| Nhánh | Nội dung |
|---|---|
| `main` | 36 run gốc (18 original + 18 active, patience=20). Không đụng. |
| `exp/active-patience50` | 18 run active mới (patience=50) + original đang chạy. Toàn bộ code mới. |

## Việc còn nợ

| | Việc | Ghi chú |
|---|---|---|
| 1 | Các mục Chương 4 còn lại: 4.2.4, 4.4.3, 4.4.4, 4.5, 5.3.1 | theo `docs/VAN_PHONG_DE_AN.md` |
| 2 | Ghép nội dung vào `De_an_thac_si_v11.docx` | chờ anh duyệt bản `Chuong4_viet_lai.docx` |
| 3 | Chốt phép kiểm: Welch hay ghép cặp | **cần ý kiến người hướng dẫn** |
| 4 | Đóng cổng 8501 | `ufw delete allow 8501/tcp` — app không có đăng nhập |
| 5 | Triển khai app qua Caddy trước khi bảo vệ | cần sửa cấu hình web đang chạy thật, phải hỏi trước |
| 6 | `topk.csv` cho tab Top-K của app | file hiện có thuộc cohort Original, chờ run mới |

## Thứ đã xong, đừng làm lại

- Bước 9: `scripts/06_make_tables.py` (Bước 9 còn thiếu `05_run_multiseed.py`, do notebook đảm nhiệm)
- Bước 10: `src/export/neo4j_export.py`, `src/export/trace.py`, `app/main.py` — `make neo4j` và `make app` chạy được
- `scripts/08_make_docx.py` — sinh Word, bảng đọc thẳng từ run
- 268 test pass (có torch cài trên VPS, nên chạy được cả test gated)
