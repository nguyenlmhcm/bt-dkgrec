# Việc đang chờ

Trạng thái sống của dự án. Cập nhật 27/08/2026.

## Đang chờ Colab — 5 run

Còn `bt_dkgrec` seed 2021, 2022 và `bt_dkgrec_l05` cả ba seed, đều trên cohort
**Original**. Khoảng 45–50 phút một run, tổng ~3,8 giờ.

Colab đã **chết một lần** sau khi xong `original/bt_dkgrec/2020` lúc 26/08 12:13
UTC và ngừng 19 tiếng. Nếu lại thấy nhịp commit ngắt quãng bất thường thì kiểm
Colab chứ đừng chờ.

Hiện có **31/36** run. Cohort Active đủ 18/18.

### Khi chạy xong

```bash
git pull
.venv/bin/python scripts/09_make_figures.py
.venv/bin/python scripts/08_make_docx.py
make tables
.venv/bin/python scripts/analysis/degree_bands.py
```

**Kiểm tái lập:** số liệu Original mới phải gần bản trên `main`. Lệch nhiều thì
điều tra trước khi dùng.

**Phân bố bậc trong 593 user được đánh giá:** bậc 1 là 259 (43,7%), bậc 2 là 76
(12,8%), bậc ≥3 là 258 (43,5%). Con số 79,6% là của 1.027.985 visitor trong
train — quần thể khác, đừng lẫn.

## Ba quyết định đang treo, cần anh chốt

| # | Quyết định | Ảnh hưởng |
|:-:|---|---|
| 1 | Ablation 2×2 tách α và λ — chạy thêm 6 run (~3,5 giờ) hay bỏ? | mục 4.2.4 |
| 2 | MBGCN/KHGT: thêm vào thư mục tham khảo, hay bỏ phát biểu "kế thừa"? | mục 3.4 + TLTK |
| 3 | Welch hay paired-by-seed làm kiểm định chính thức? | cần người hướng dẫn |

Chi tiết quyết định 2: thư mục tham khảo của v11 có **18 mục, không có MBGCN
cũng không có KHGT**, nhưng D30 ghi α và λ kế thừa từ hai paper đó. Phát biểu
kế thừa mà không trích dẫn là điểm yếu. Hướng trung thực hơn là thêm trích dẫn.

## Đính chính lớn ngày 27/08 — đã sửa, ghi lại để không lặp

Lập luận "với user bậc 1, trọng số bị triệt tiêu" là **SAI**. Đó là hệ quả của
chuẩn hoá **theo hàng** `D⁻¹A`. Mô hình dùng dạng **đối xứng**
`D^(−1/2) A D^(−1/2)`, nơi mẫu số là `√dᵤ`:

```
â = W / (√W · √dᵢ) = √W / √dᵢ
```

Trọng số vào theo **căn bậc hai**. Tỉ lệ 3:1 giữa `transaction` và `view` còn
`√3 ≈ 1,73:1` — nhẹ hơn danh nghĩa nhưng **không** triệt tiêu, và tác động trên
mọi dải bậc.

Bị bác bỏ bởi chính tiên đoán đã phát biểu trước: `warm_deg1` phải lệch đúng 0,
đo được lệch +0.00386. Đã sửa ở 12 chỗ. Test `test_patch_chuong3.py` chặn các
chuỗi `"bị triệt tiêu"`, `"triệt tiêu hoàn toàn"`, `"79.6%"` quay lại file .docx.

## Còn nợ ở phần viết

- Chương 4: mục **4.2.4** (ablation), **4.4.3** (phân tích lỗi), **4.4.4** và
  **5.3.1** (hạn chế), **4.5** (kiểm chứng ứng dụng)
- Chương 4 và 5 **sinh lại bằng script**, không vá — Ch4 có 1 công thức OMML,
  Ch5 có 0. Chỉ Chương 3 phải vá (33 công thức).
- Tóm tắt và Summary chứa số cũ, phải viết lại
- Khi trình bày: `learning_rate = 0.005` và `batch_size = 65.536` **không** lấy
  từ paper LightGCN (paper dùng 0.001 và 1.024) mà kế thừa từ v11. Chỉ
  `embedding_dim`, `num_layers`, hệ số L2 là theo paper. Đừng nói gộp "theo paper".

## Việc vặt

- **Cổng 8501 vẫn mở ra internet**, app không có đăng nhập. Đóng bằng
  `ufw delete allow 8501/tcp` khi xem xong.
- Triển khai app qua Caddy trước buổi bảo vệ — đụng cấu hình site đang chạy, hỏi trước.
- Consensus MCP hết quota tháng (30/30), reset 01/09. Dùng WebSearch thay thế.

## Đã xong, đừng làm lại

| Việc | Kết quả |
|---|---|
| Hình Chương 4 | 4 hình đúng bố cục v11, sinh từ `src/evaluation/figures.py` |
| Vá Chương 3 | `docs/De_an_thac_si_v12.docx`, 6 mục, tô nền vàng + danh mục sửa đổi ở trang đầu |
| Bản nộp sạch | `scripts/10_patch_chuong3.py --no-highlight` |
| Trang kiểm kê | `docs/KIEM_KE.html` + artifact |
| Trang mạch đề án | `docs/MACH_DE_AN.html` + artifact — nền tảng, ba paper, mạch, 11 câu bảo vệ |
| Sửa test gãy trên Colab | `pytest.importorskip("docx")` ở 2 chỗ; python-docx đã khai vào requirements.txt |
| Quy ước số | metric dùng dấu chấm, phần trăm dùng dấu phẩy — xem `VAN_PHONG_DE_AN.md` |

Test: **299 pass trên VPS**, **273 pass + 2 skipped khi giả lập Colab** (chặn
module `docx`).
