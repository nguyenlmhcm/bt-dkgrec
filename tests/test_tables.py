"""Tests for Buoc 9 — sinh bang ket qua (`scripts/06_make_tables.py`).

Buoc nay bien `experiments/runs/` thanh nguon su that duy nhat cho moi bang
trong de an, nen no phai that bai TO va SOM khi dau vao khong sach, chu khong
duoc in ra mot bang trong sai am tham.

Torch-free co chu y: bang duoc sinh tren VPS noi khong co torch (D28).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tables_script():
    """Import `scripts/06_make_tables.py` theo duong dan (`scripts/` khong phai package)."""
    spec = importlib.util.spec_from_file_location(
        "tables_script", REPO_ROOT / "scripts" / "06_make_tables.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _write_run(root: Path, cohort: str, model: str, seed: int, value: float,
               stamp: str = "20260101-000000") -> Path:
    """Mot run artifact toi thieu, du cho `load_runs()` doc duoc."""
    run = root / f"{cohort}_{model}_{seed}_{stamp}"
    run.mkdir(parents=True)
    warm = {"recall@20": value, "ndcg@20": value / 2,
            "hit_rate@20": value * 2, "coverage@20": 0.01}
    band = dict(warm)
    (run / "metrics.json").write_text(json.dumps({
        "run_id": run.name, "cohort": cohort, "model": model, "seed": seed,
        "test": {
            "n_users": {"warm": 100, "warm_deg1": 60, "warm_deg2": 20,
                        "warm_deg3plus": 20, "cold": 0, "all": 100},
            "warm": warm, "warm_deg1": band, "warm_deg2": band,
            "warm_deg3plus": band, "cold": None, "all": None,
        },
    }), encoding="utf-8")
    return run


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    for seed, value in zip((2020, 2021, 2022), (0.020, 0.022, 0.024)):
        _write_run(root, "original", "bt_dkgrec", seed, value)
    for seed, value in zip((2020, 2021, 2022), (0.018, 0.019, 0.021)):
        _write_run(root, "original", "static_kg_gcn", seed, value)
    for seed, value in zip((2020, 2021, 2022), (0.015, 0.016, 0.017)):
        _write_run(root, "original", "lightgcn", seed, value)
    return root


# ── Guard chong trung o ──────────────────────────────────────────────────


def test_a_duplicated_cell_stops_the_whole_run(runs_dir) -> None:
    """★ Hai run cho cung (cohort, model, seed) se bi lay trung binh am tham.

    Day la cach mot ngan sach dung som cu tron voi ngan sach moi ma bang van in
    ra binh thuong. Phai chet, khong duoc doan.
    """
    module = _tables_script()
    _write_run(runs_dir, "original", "bt_dkgrec", 2020, 0.999, stamp="20260202-000000")

    from src.evaluation.reporting import load_runs
    frame = load_runs(runs_dir, "test", "warm")

    with pytest.raises(SystemExit) as excinfo:
        module.assert_no_duplicate_cells(frame)
    assert "original/bt_dkgrec/2020" in str(excinfo.value)


def test_clean_input_passes_the_duplicate_guard(runs_dir) -> None:
    module = _tables_script()
    from src.evaluation.reporting import load_runs

    module.assert_no_duplicate_cells(load_runs(runs_dir, "test", "warm"))  # khong nem


def test_the_duplicate_guard_accepts_an_empty_frame() -> None:
    import pandas as pd
    module = _tables_script()

    module.assert_no_duplicate_cells(pd.DataFrame())  # khong nem


# ── Phep kiem ghep cap ───────────────────────────────────────────────────


def test_paired_test_counts_wins_and_pairs(runs_dir) -> None:
    module = _tables_script()
    from src.evaluation.reporting import load_runs

    frame = load_runs(runs_dir, "test", "warm")
    result = module.paired_test(frame, "bt_dkgrec", "static_kg_gcn", "original", "recall@20")

    assert result["n"] == 3
    assert result["wins"] == 3          # bt_dkgrec cao hon o ca ba seed
    assert 0.0 <= result["p_value"] <= 1.0


def test_paired_test_refuses_a_single_seed(tmp_path) -> None:
    module = _tables_script()
    from src.evaluation.reporting import load_runs

    root = tmp_path / "runs"
    root.mkdir()
    _write_run(root, "original", "bt_dkgrec", 2020, 0.02)
    _write_run(root, "original", "static_kg_gcn", 2020, 0.01)
    frame = load_runs(root, "test", "warm")

    assert module.paired_test(frame, "bt_dkgrec", "static_kg_gcn",
                              "original", "recall@20")["p_value"] is None


def test_paired_test_refuses_a_constant_difference(tmp_path) -> None:
    """Hieu so khong doi -> phuong sai 0 -> t vo cuc. Bao None thay vi p=0."""
    module = _tables_script()
    from src.evaluation.reporting import load_runs

    root = tmp_path / "runs"
    root.mkdir()
    for seed, value in zip((2020, 2021, 2022), (0.020, 0.022, 0.024)):
        _write_run(root, "original", "bt_dkgrec", seed, value)
        _write_run(root, "original", "static_kg_gcn", seed, value - 0.005)
    frame = load_runs(root, "test", "warm")

    assert module.paired_test(frame, "bt_dkgrec", "static_kg_gcn",
                              "original", "recall@20")["p_value"] is None


# ── Bang sinh ra ─────────────────────────────────────────────────────────


def test_the_comparison_table_always_flags_the_paired_column_as_unsettled(runs_dir) -> None:
    """Doi phep kiem sau khi nhin ket qua ma khong noi la sai. Chu thich bat buoc."""
    module = _tables_script()
    from src.evaluation.reporting import load_runs

    table = module.comparison_table(load_runs(runs_dir, "test", "warm"),
                                    "original", "recall@20")
    assert "CHUA duoc chot" in table
    assert "Welch" in table


def test_the_core_comparison_comes_first(runs_dir) -> None:
    """Dong vs tinh la cap chung minh chu "dong" trong ten de tai."""
    module = _tables_script()

    assert module.COMPARISONS[0] == ("bt_dkgrec", "static_kg_gcn")
