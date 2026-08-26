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

**Con số cần nhìn:** `warm_deg3plus` trên cohort Original (~50 user trong 593).
Đây là nhóm duy nhất mà behavior-time weighting có thể tác động — 79,6% user chỉ
có 1 cạnh nên trọng số bị chuẩn hoá triệt tiêu đúng bằng 0.

**Tiên đoán đã phát biểu trước:** `warm_deg1` phải cho hiệu số **đúng bằng 0**
giữa `bt_dkgrec` và `static_kg_gcn`. Khác 0 nghĩa là code sai.

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
