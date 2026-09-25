# Việc đang chờ

Cập nhật: 21/09/2026. Hạn nộp: giữa tháng 9/2026.

---

## Trạng thái: file Word ở bản v17, CHƯA COMMIT

| File | Dùng để |
|---|---|
| `docs/De_an_thac_si_v17.docx` | rà soát — tô vàng chỗ vừa sửa, còn trang nhật ký sửa đổi |
| **`docs/De_an_thac_si_v17_ban_nop.docx`** | **nộp** — 0 tô nền, 0 trang nhật ký, trang bìa còn nguyên |

Bản nộp: 46 bảng · 10 hình · 23 công thức · `±` 0 · `giá trị p` 0 · `Welch` 0.
Tóm tắt: 451 từ, 2 con số, 0 giá trị thập phân (v15: 478 / 25 / 8). pytest **354 xanh**.

**8 file chưa commit:** `docs/De_an_thac_si_v16*.docx`, `docs/De_an_thac_si_v17*.docx`,
`scripts/14_bang_seed.py`, `scripts/15_bo_pm.py`, `tests/test_bang_seed.py`, `tests/test_bo_pm.py`.

---

## Chuỗi vá (chạy tay, mỗi script nhận .docx của bước trước)

| Bước | Script | Làm gì |
|---|---|---|
| v11→v12 | `10_patch_chuong3.py` | vá Chương 3 |
| v12→v13 | `11_patch_chuong12.py` | vá Chương 1–2 |
| v13→v14 | `12_cat_cong_thuc.py` | cắt 13 công thức, đánh số lại |
| v14→v15 | `13_hoan_thien.py` | ghép Chương 4 sinh lại |
| v15→v16 | `14_bang_seed.py` | **Bảng 4.7** kết quả từng seed; sửa đánh số bảng C4 (thiếu 4.3, có 4.4b, 4.8 dùng 2 lần); viết lại Tóm tắt/Summary |
| v16→v17 | `15_bo_pm.py` | bỏ `±`; xoá Bảng 4.9 (p-value); gỡ cột Welch/Ghép cặp; **Bảng 4.8 đổi sang λ=0.05**; viết lại 4 đoạn dẫn; `--bo-changelog` gỡ trang nhật ký cho bản nộp |

Bản nộp sinh bằng: `python scripts/15_bo_pm.py --no-highlight --bo-changelog --out docs/De_an_thac_si_v17_ban_nop.docx`

---

## Kết quả cuối cùng — Recall@20, trung bình 3 seed

| Mô hình | Original (593) | Active (234) |
|---|---|---|
| LightGCN | 0.023736 | 0.028601 |
| Static KG-GCN | 0.027178 | 0.031317 |
| BT-DKGRec λ=0.01 | 0.028778 | 0.032445 |
| **BT-DKGRec λ=0.05** | **0.030453** | **0.038004** |

**Mô hình đề xuất = λ=0.05** (đã dò trên tập xác thực, Bảng 3.6). Vượt LightGCN **+28,3%** / **+32,9%**.
Từng seed (Bảng 4.7): lần chạy thấp nhất của λ=0.05 (0.029146 / 0.033901) vẫn cao hơn lần chạy
cao nhất của LightGCN (0.025990 / 0.031293). **Chỉ đúng với λ=0.05** — λ=0.01 chồng lấn (0.023102 < 0.025990).

Mức cải thiện theo chỉ số dao động +7,5% (Coverage, active) đến +32,9% (Recall, active). NDCG chỉ thắng 2/3 seed.

---

## Giọng văn — bằng chứng đo trên hai bài mẫu HUIT

| | Khang | Tuyến | Của mình (v15 → v17) |
|---|---|---|---|
| `p-value` | 0 | 0 | có → **0** |
| `±` | 0 | 0 | 75 → **0** |
| Số thập phân trong Tóm tắt | 0 | 0 | 8 → **0** |

**Đừng gọi skill `ars-abstract` / `academic-paper` cho việc văn phong nữa.** Đã thử: nó nhắm bài báo
tạp chí zh-TW/EN theo APA 7.0 — chuẩn *bắt buộc* `M ± SD` và `p`. Không có dòng nào về `±`.
Chuẩn chi phối là khung HUIT + hai bài mẫu.

---

## Còn phải làm

1. **Commit 8 file** (chưa làm, anh Nguyên chưa bảo)
2. **Rà v17 trong Word** — Ctrl+A, F9 cập nhật mục lục
3. **Cụm "ứng dụng nguyên mẫu"** còn ở tiêu đề §4.5 và caption Bảng 4.12 — anh đã chê "nguyên mẫu cục bộ"
   trong Tóm tắt; sửa tiêu đề là đụng mục lục nên chờ anh gật
4. **Đóng port 8501** — `ufw delete allow 8501/tcp`, app Streamlit mở ra internet không có đăng nhập
5. Quyết định treo: ablation tách α khỏi λ (6 run, ~3,5 giờ) — hiện §4.2.4 nói thẳng α chưa dò, nộp được

---

## Bẫy đã sập, đừng lặp lại

- **Trang nhật ký sửa đổi đứng TRƯỚC trang bìa.** Gỡ nó mà đi tới tiêu đề kế là nuốt luôn hai bảng
  "BỘ CÔNG THƯƠNG…" + logo trường. `15_bo_pm.py::xoa_muc_changelog` dừng đúng sau bảng có ô "Mục".
  Test `test_ban_nop_van_con_trang_bia` chốt.
- **Bảng danh mục sửa đổi trích lại thứ vừa gỡ** (`±`, `4.4b`, `Welch`). Mọi phép kiểm "không còn X"
  phải loại bảng đó ra, nếu không là dương tính giả.
- **Caption trong mục lục (List Bullet) trùng chữ với caption thân bài.** `para_khop` phải lọc style.
- **Tóm tắt và Bảng 4.8 phải cùng một λ.** v16 từng nói 28%/33% ở Tóm tắt nhưng +21%/+13% ở Bảng 4.8.
  `test_bang_48_khop_con_so_trong_tom_tat` chốt.

---

## Ba câu trả lời khó, đã chuẩn bị sẵn

**"Sao chọn RetailRocket?"** — theo yêu cầu mô hình: nhiều loại hành vi, timestamp từng sự kiện,
thuộc tính sản phẩm, cây danh mục. Độ thưa 3,33% là bài toán, không phải khiếm khuyết.

**"Batch 65.536 ở đâu ra?"** — code lan truyền lại toàn đồ thị ở mỗi batch (`trainer.py:331` trong
vòng lặp batch). Batch lớn để giảm số lần lan truyền: 170 epoch mất 32 phút; batch 1.024 sẽ là ~34 giờ
× 36 run ≈ 71 ngày. Mặc định model là 2.048; 65.536 do wrapper Kaggle ghi đè — quyết định hạ tầng.

**"30% có tin được không?"** — Bảng 4.7: ba lần chạy của mô hình đề xuất đều nằm trên ba lần chạy
của LightGCN, không lần nào chồng lấn. Và nói rõ 28% là ở Recall@20; các chỉ số khác từ +7,5% đến +31%.
