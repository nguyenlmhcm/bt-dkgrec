"""Rang buoc giua script sinh hinh va script ghi Word.

Viec ve hinh da co `tests/test_reporting.py` kiem (mau, hatch, nhan gia tri).
Cho con lai chua ai giu la **ten file**: `09_make_figures.py` ve ra mot ten,
`08_make_docx.py` di tim mot ten. Neu hai ten lech nhau, Word van ghi ra binh
thuong, chi thay dong "[Thieu hinh ...]" o giua chuong -- hong am tham. Test o
day chot rang ca hai cung tra cuu qua `figure_path()`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.evaluation.figures import FIGURE_NAMES, figure_path

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"),
                                                  ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_figure_path_covers_the_two_chapter_four_figures():
    assert {"metrics", "loss"} <= set(FIGURE_NAMES)


def test_metrics_and_loss_paths_differ_per_cohort():
    """Original va active khong duoc ghi de len hinh cua nhau."""
    out = Path("/tmp/x")
    paths = {figure_path(out, kind, cohort)
             for kind in ("metrics", "loss")
             for cohort in ("original", "active")}
    assert len(paths) == 4


def test_unknown_figure_kind_raises_rather_than_guessing():
    with pytest.raises(KeyError):
        figure_path(Path("/tmp/x"), "khong-ton-tai", "active")


def test_the_two_scripts_agree_on_where_figures_live():
    """Rang buoc chinh: ca hai script dung cung mot ham va cung mot thu muc."""
    maker = _script("09_make_figures.py")
    docx = _script("08_make_docx.py")
    assert maker.figures.figure_path is figure_path
    assert docx.figure_path is figure_path
    assert docx.FIGURES == Path("docs/figures")


def test_the_maker_does_not_reimplement_the_palette():
    """`09` phai la lop dieu khien; dinh nghia mau nam o `src/evaluation/figures.py`.

    Mot ban sao MODEL_COLORS trong script se troi khoi ban goc va Chuong 4 se co
    hai he mau cho cung mot mo hinh.
    """
    source = (ROOT / "scripts" / "09_make_figures.py").read_text(encoding="utf-8")
    assert "MODEL_COLORS" not in source
    assert "MODEL_HATCH" not in source
