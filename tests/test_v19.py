"""v18 -> v19: quyen mo ta dung viec ung dung doc CSV, khong ket noi Neo4j."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("docx", reason="can python-docx — chi chay tren VPS")

from docx import Document  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V18 = ROOT / "docs" / "De_an_thac_si_v18.docx"

pytestmark = pytest.mark.skipif(not V18.is_file(), reason="chua co ban v18")


@pytest.fixture(scope="module")
def nop(tmp_path_factory) -> Document:
    spec = importlib.util.spec_from_file_location("v19", ROOT / "scripts" / "20_v19.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v19"] = mod
    spec.loader.exec_module(mod)
    out = tmp_path_factory.mktemp("nop") / "v19.docx"
    sys.argv = ["20_v19.py", "--out", str(out), "--no-highlight", "--bo-changelog"]
    mod.main()
    return Document(str(out))


def _text(doc: Document) -> str:
    return "\n".join(p.text for p in doc.paragraphs)


@pytest.mark.parametrize("cu", ["ứng dụng nguyên mẫu để truy vấn Neo4j", "sử dụng Neo4j làm kho lưu trữ và truy vấn"])
def test_khong_con_noi_ung_dung_truy_van_neo4j(nop, cu):
    assert cu not in _text(nop)


@pytest.mark.parametrize("cu", ["lưu vết", "lớp chiếu", "đồ thị chiếu", "Đồ thị chiếu", "graph huấn luyện"])
def test_chi_con_hai_ten_do_thi(nop, cu):
    o = [c.text for t in nop.tables for r in t.rows for c in r.cells]
    assert cu not in _text(nop) and not any(cu in x for x in o)


def test_lop_thong_tin_khong_bi_doi_nham(nop):
    assert "kết hợp hai lớp thông tin" in _text(nop)


def test_doan_co_cong_thuc_con_nguyen(nop):
    p = [x for x in nop.paragraphs if x.text.startswith("Quan hệ ") and "đồ thị truy vết trả lời" in x.text]
    assert len(p) == 1 and p[0]._p.find(".//" + qn("m:oMath")) is not None


def test_cau_mo_ta_buoc_xuat_van_con(nop):
    """Doan nay von da dung voi code; v19 khong duoc dong vao."""
    assert "vừa là nguồn đọc trực tiếp của ứng dụng nguyên mẫu" in _text(nop)


def test_ung_dung_doc_csv_khong_nhap_driver_neo4j():
    """Neu sau nay app ket noi Neo4j that thi phai sua lai quyen, test nay se bao."""
    for f in (ROOT / "app" / "main.py", ROOT / "src" / "export" / "trace.py"):
        code = f.read_text(encoding="utf-8")
        assert "GraphDatabase" not in code and "import neo4j" not in code


def test_hinh_3_1_la_anh_moi_va_ban_nop_sach(nop):
    moi = hashlib.sha1((ROOT / "docs" / "figures" / "fig_v19_3_1_kien_truc.png").read_bytes()).hexdigest()
    anh = [hashlib.sha1(nop.part.rels[b.get(qn("r:embed"))].target_part.blob).hexdigest()
           for b in nop.element.body.iter(qn("a:blip"))]
    assert moi in anh and len(anh) == 10
    assert not any(r.font.highlight_color for p in nop.paragraphs for r in p.runs)
    assert not any("DANH MỤC SỬA ĐỔI" in p.text for p in nop.paragraphs)
