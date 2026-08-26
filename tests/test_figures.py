"""Kiem tra script sinh hinh Chuong 4.

Diem quan trong nhat khong phai la hinh "dep", ma la hinh KHONG duoc noi khac
bang so nam ngay tren no. Vi vay test kiem chieu cao cot ve tu du lieu, chu
khong chi kiem file PNG co ton tai.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "make_figures", ROOT / "scripts" / "09_make_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(runs_dir: Path, cohort: str, model: str, seed: int,
              recall: float, losses: list[float] | None,
              stamp: str = "20260101-000000") -> None:
    """Mot run artifact toi thieu -- cung khuon voi `tests/test_tables.py`."""
    run = runs_dir / f"{cohort}_{model}_{seed}_{stamp}"
    run.mkdir(parents=True)
    warm = {"recall@20": recall, "ndcg@20": recall / 2,
            "hit_rate@20": recall * 2, "coverage@20": 0.01}
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
    rows = []
    if losses is None:  # heuristic: khong hoc, khong co loss
        rows.append({"model": model, "cohort": cohort, "seed": seed, "epoch": 0,
                     "loss": None, "note": "mo hinh khong hoc"})
    else:
        for epoch, loss in enumerate(losses, start=1):
            rows.append({"model": model, "cohort": cohort, "seed": seed,
                         "epoch": epoch, "loss": loss, "note": ""})
    pd.DataFrame(rows).to_csv(run / "curves.csv", index=False)


@pytest.fixture()
def runs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    write_run(d, "active", "popularity", 2020, 0.01, None)
    write_run(d, "active", "lightgcn", 2020, 0.02, [0.5, 0.3, 0.2])
    write_run(d, "active", "lightgcn", 2021, 0.04, [0.6, 0.35, 0.25])
    return d


def test_bar_height_matches_the_table(runs_dir, tmp_path):
    """Chieu cao cot phai bang dung con so trong bang, khong phai gan bang.

    Doc thang tu `ax.patches` sau khi ve, chu khong tinh lai trung binh roi so
    voi chinh no -- nguoc lai thi test se pass ke ca khi ham ve dung sai cot.
    """
    module = load_module()
    from src.evaluation.reporting import load_runs

    frame = load_runs(runs_dir, split="test", segment="warm")
    subset = frame[frame["cohort"] == "active"]
    out = tmp_path / "active_metrics.png"
    module.figure_metrics(frame, "active", out)
    assert out.exists() and out.stat().st_size > 0

    # popularity: mot seed 0.01; lightgcn: hai seed 0.02 va 0.04 -> 0.03
    means, stds = module.bar_values(subset, "lightgcn")
    assert means[0] == pytest.approx(0.03)
    assert stds[0] == pytest.approx(subset[subset["model"] == "lightgcn"]
                                    ["recall@20"].std(ddof=1))
    assert module.bar_values(subset, "popularity") == ([0.01, 0.005, 0.02], [0.0, 0.0, 0.0])


def test_drawn_bars_are_exactly_the_bar_values(runs_dir, tmp_path):
    """Kiem chinh cac hinh chu nhat da ve, khong kiem ham tinh."""
    import matplotlib.pyplot as plt

    module = load_module()
    from src.evaluation.reporting import load_runs

    frame = load_runs(runs_dir, split="test", segment="warm")
    subset = frame[frame["cohort"] == "active"]
    models = [m for m in module.MODEL_ORDER if m in set(subset["model"])]

    drawn = []
    real_subplots = plt.subplots

    def capture(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        drawn.append(ax)
        return fig, ax

    plt.subplots = capture
    try:
        module.figure_metrics(frame, "active", tmp_path / "x.png")
    finally:
        plt.subplots = real_subplots

    heights = [round(patch.get_height(), 6) for patch in drawn[0].patches]
    expected = [round(v, 6) for m in models for v in module.bar_values(subset, m)[0]]
    assert heights == expected


def test_heuristics_are_dropped_from_the_loss_figure(runs_dir, tmp_path):
    """Popularity khong co duong hoi tu; no phai bi loai chu khong ve thanh 0."""
    module = load_module()
    curves = module.load_curves(runs_dir, "active")
    assert set(curves["model"]) == {"lightgcn"}
    out = tmp_path / "active_loss.png"
    module.figure_loss(curves, "active", out)
    assert out.exists()


def test_every_model_has_a_fixed_colour(runs_dir):
    """Mau gan theo mo hinh, khong cap phat theo thu tu -- Hinh 4.1 va 4.3 phai
    dung cung mot mau cho cung mot mo hinh."""
    module = load_module()
    from src.evaluation.reporting import MODEL_ORDER

    assert set(MODEL_ORDER) <= set(module.MODEL_COLORS)
    assert len(set(module.MODEL_COLORS.values())) == len(module.MODEL_COLORS)


def test_duplicate_runs_stop_the_figures(runs_dir, tmp_path):
    """Cung mot guard voi bang: hai run trung o thi dung, khong ve am tham."""
    from src.evaluation.reporting import assert_no_duplicate_cells, load_runs

    write_run(runs_dir, "active", "lightgcn", 2020, 0.99, [0.5],
              stamp="20260202-000000")
    with pytest.raises(SystemExit):
        assert_no_duplicate_cells(load_runs(runs_dir, split="test", segment="warm"))


def test_missing_cohort_raises_rather_than_drawing_an_empty_axis(tmp_path):
    module = load_module()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        module.load_curves(empty, "original")
