"""v20 -> v21: quyen mo ta dung hai trang cua ung dung, khong con noi chua suy luan."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("docx", reason="can python-docx — chi chay tren VPS")

from docx import Document  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V20 = ROOT / "docs" / "De_an_thac_si_v20.docx"
pytestmark = pytest.mark.skipif(not V20.is_file(), reason="chua co ban v20")


@pytest.fixture(scope="module")
def nop(tmp_path_factory) -> Document:
    spec = importlib.util.spec_from_file_location("v21", ROOT / "scripts" / "24_v21.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v21"] = mod
    spec.loader.exec_module(mod)
    out = tmp_path_factory.mktemp("nop") / "v21.docx"
    sys.argv = ["24_v21.py", "--out", str(out), "--no-highlight", "--bo-changelog"]
    mod.main()
    return Document(str(out))


def _all(doc: Document) -> str:
    cells = [c.text for t in doc.tables for r in t.rows for c in r.cells]
    return "\n".join([p.text for p in doc.paragraphs] + cells)


@pytest.mark.parametrize("cu", [
    "Ứng dụng chưa nạp trạng thái", "kho mã nguồn chưa lưu trạng thái", "Chưa nạp trạng thái mô hình vào ứng dụng",
    "chưa bao gồm dịch vụ suy luận", "phần trình diễn không phụ thuộc",
])
def test_khong_con_noi_chua_suy_luan(nop, cu):
    assert cu not in _all(nop)


def test_noi_ro_gioi_han_cua_trang_mo_phong(nop):
    text = _all(nop)
    assert "không thuộc các bảng kết quả" in text and "chưa tự động huấn luyện lại" in text


def test_ung_dung_that_su_co_du_hai_trang():
    main = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    page = (ROOT / "app" / "pages" / "1_Mo_phong_online.py").read_text(encoding="utf-8")
    assert "load_topk" in main and "GraphStore" in page and "service.recommend" in page
