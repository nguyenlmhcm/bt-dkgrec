"""v17 -> v18: han che dung su that, so lieu D35 doc tu runs, ban gui co sach."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("docx", reason="can python-docx — chi chay tren VPS")

from docx import Document  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V17 = ROOT / "docs" / "De_an_thac_si_v17.docx"
RUNS = ROOT / "experiments" / "runs"

pytestmark = pytest.mark.skipif(not V17.is_file(), reason="chua co ban v17")


def _load():
    spec = importlib.util.spec_from_file_location("v18", ROOT / "scripts" / "17_v18.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["v18"] = module
    spec.loader.exec_module(module)
    return module


def _build(tmp_path: Path, *extra: str) -> Path:
    mod = _load()
    out = tmp_path / "v18.docx"
    sys.argv = ["17_v18.py", "--out", str(out), *extra]
    mod.main()
    return out


@pytest.fixture(scope="module")
def nop(tmp_path_factory) -> Document:
    return Document(str(_build(tmp_path_factory.mktemp("nop"), "--no-highlight", "--bo-changelog")))


@pytest.fixture(scope="module")
def ra_soat(tmp_path_factory) -> Document:
    return Document(str(_build(tmp_path_factory.mktemp("rs"))))


def _text(doc: Document) -> str:
    return "\n".join(p.text for p in doc.paragraphs)


@pytest.mark.parametrize("cu", ["một seed (2026)", "Kết quả hiện dùng một seed",
                                "10 epoch chưa đủ", "nguyên mẫu cục bộ"])
def test_han_che_cu_cua_v11_khong_con(nop, cu):
    assert cu not in _text(nop)


def test_so_lieu_d35_khop_runs(nop):
    """Moi so trong doan moi phai bang trung binh doc tu metrics.json."""
    so = _load().doc_so(RUNS)["mean"]
    text = _text(nop)
    for cohort in ("original", "active"):
        for model in ("bt_dkgrec_l05", "static_kg_gcn", "bt_dkgrec_alpha_only", "bt_dkgrec_time_only"):
            assert f"{so[(cohort, model)]:.6f}" in text, (cohort, model)


def test_doan_424_khong_con_so_cua_lambda_001(nop):
    p = [x.text for x in nop.paragraphs
         if x.text.startswith("Bỏ trọng số hành vi-thời gian khỏi mô hình đề xuất")]
    assert len(p) == 1 and "0.028778" not in p[0] and "λ=0.05" in p[0]


def test_ban_nop_sach_va_con_trang_bia(nop):
    assert not any(r.font.highlight_color for p in nop.paragraphs for r in p.runs)
    assert not any("DANH MỤC SỬA ĐỔI" in p.text for p in nop.paragraphs)
    assert any("BỘ CÔNG THƯƠNG" in c.text for t in nop.tables for r in t.rows for c in r.cells)


def test_ban_ra_soat_chi_to_cho_vua_sua(ra_soat):
    """To nen cu cua v17 da bi xoa: moi doan duoc to phai la doan v18 vua sua."""
    to = [p.text for p in ra_soat.paragraphs if any(r.font.highlight_color for r in p.runs)]
    # 7 doan chu vua sua + 3 chu thich Hinh 3.1-3.3 ve lai.
    assert 1 <= len(to) <= 10, to
    assert any("[v18]" in c.text for t in ra_soat.tables for r in t.rows for c in r.cells)


def test_moi_hinh_deu_co_anh_that(nop):
    """Tu v15 sau hinh Chuong 4 tro vao khong khi va khong test nao bat. Chan lai."""
    from docx.oxml.ns import qn

    rels = nop.part.rels
    blips = list(nop.element.body.iter(qn("a:blip")))
    hong = [b.get(qn("r:embed")) for b in blips if b.get(qn("r:embed")) not in rels]
    assert len(blips) == 10 and not hong, hong


def test_hinh_chuong_3_la_anh_ve_lai(nop):
    """Hinh 3.1-3.3 phai la anh do scripts/18 sinh, khong con anh v11."""
    import hashlib

    from docx.oxml.ns import qn

    moi = {hashlib.sha1(p.read_bytes()).hexdigest()
           for p in (ROOT / "docs" / "figures").glob("fig_3_*.png")}
    trong_quyen = {hashlib.sha1(nop.part.rels[b.get(qn("r:embed"))].target_part.blob).hexdigest()
                   for b in nop.element.body.iter(qn("a:blip"))}
    assert len(moi) == 3 and moi <= trong_quyen
