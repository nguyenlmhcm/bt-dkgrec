"""v21 -> v22: quyen ta dung ba phan he, khong con cau lech bang so, Bang 4.9 dung lambda=0.05."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("docx", reason="can python-docx — chi chay tren VPS")

from docx import Document  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOP = ROOT / "docs" / "De_an_thac_si_v22_ban_nop.docx"
pytestmark = pytest.mark.skipif(not NOP.is_file(), reason="chua co ban v22")


@pytest.fixture(scope="module")
def doc() -> Document:
    return Document(str(NOP))


def _all(doc: Document) -> str:
    cells = [c.text for t in doc.tables for r in t.rows for c in r.cells]
    return "\n".join([p.text for p in doc.paragraphs] + cells)


@pytest.mark.parametrize("cu", [
    "v11", "mô phỏng online", "hai trang", "web cục bộ", "bộ dữ liệu mẫu", "cạnh tranh với LightGCN",
    "Thứ tự giữa các mô hình giữ nguyên", "Document/figures", "ý nghĩa thống kê thông thường", "λ=0,",
    "lợi thế rõ nhất chỉ xuất hiện", "cùng danh mục với những sản phẩm",
])
def test_khong_con_chu_cu(doc, cu) -> None:
    assert cu not in _all(doc)


def test_so_doi_tuong(doc) -> None:
    assert len(doc.element.body.findall(".//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath")) == 23
    assert len(doc.tables) == 46
    assert len(doc.inline_shapes) == 13


def test_danh_muc_co_ba_hinh_moi(doc) -> None:
    for so in ("Hình 4.7.", "Hình 4.8.", "Hình 4.9."):
        assert sum(p.text.startswith(so) for p in doc.paragraphs) == 2


def test_ban_nop_khong_to_nen(doc) -> None:
    assert not any(r.font.highlight_color for p in doc.paragraphs for r in p.runs)


def test_bang_4_9_la_lambda_005(doc) -> None:
    bang = next(t for t in doc.tables if len(t.columns) == 6 and t.rows[0].cells[2].text == "Static KG-GCN")
    assert bang.rows[0].cells[3].text == "BT-DKGRec-GCN (λ=0.05)"
    assert [c.text for c in bang.rows[1].cells[2:]] == ["0.027178", "0.030453", "+12,05%", "2/3"]
    assert [c.text for c in bang.rows[2].cells[4:]] == ["+11,28%", "1/3"]
