"""v19 -> v20: cau ve hoi tu doc tu curves.csv, danh muc khop chu thich, Hinh 3.1 theo thiet ke."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("docx", reason="can python-docx — chi chay tren VPS")

from docx import Document  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V19 = ROOT / "docs" / "De_an_thac_si_v19.docx"

pytestmark = pytest.mark.skipif(not V19.is_file(), reason="chua co ban v19")


def _mod():
    spec = importlib.util.spec_from_file_location("v20", ROOT / "scripts" / "21_v20.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v20"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def nop(tmp_path_factory) -> Document:
    mod = _mod()
    out = tmp_path_factory.mktemp("nop") / "v20.docx"
    sys.argv = ["21_v20.py", "--out", str(out), "--no-highlight", "--bo-changelog"]
    mod.main()
    return Document(str(out))


def _text(doc: Document) -> str:
    return "\n".join(p.text for p in doc.paragraphs)


def test_cau_hoi_tu_cu_da_het_va_so_khop_curves(nop):
    mod = _mod()
    so = mod.doc_hoi_tu(mod.RUNS)
    text = _text(nop)
    assert "huấn luyện đến khi kết quả trên tập xác thực không còn cải thiện" not in text
    assert mod.cau_hoi_tu(so) in text
    assert sum(cuoi >= mod.TRAN_EPOCH for _, cuoi, _ in so["active"]) == 10
    assert "riêng một lần chạy của mô hình λ=0.05 ở epoch 400" in text


def test_cau_hoi_tu_tu_choi_khi_original_cham_tran():
    mod = _mod()
    so = {"original": [("x", 1000, 900)], "active": [("x", 1000, 900)]}
    with pytest.raises(SystemExit):
        mod.cau_hoi_tu(so)


def test_danh_muc_khop_chu_thich(nop):
    caps, dm = {}, {}
    for p in nop.paragraphs:
        m = re.match(r"^((?:Hình|Bảng) \d+\.\d+)\.", p.text.strip())
        if m:
            (dm if p.style.name == "List Bullet" else caps)[m.group(1)] = p.text.strip()
    assert len(dm) == 30 and dm == caps


def test_hinh_3_1_moi_co_trong_quyen_va_tep_chinh_tay(nop):
    png = ROOT / "docs" / "figures" / "fig_v20_3_1_kien_truc.png"
    moi = hashlib.sha1(png.read_bytes()).hexdigest()
    anh = [hashlib.sha1(nop.part.rels[b.get(qn("r:embed"))].target_part.blob).hexdigest()
           for b in nop.element.body.iter(qn("a:blip"))]
    assert moi in anh and len(anh) == 10
    assert "Nét đứt" in png.with_suffix(".drawio").read_text(encoding="utf-8")
    assert not any(r.font.highlight_color for p in nop.paragraphs for r in p.runs)
