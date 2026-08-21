"""Tests for run aggregation, tables and figures.

These protect the integrity rules: the table reads every run, an unmeasured
segment is never counted as zero, and the deterministic-baseline footnote is
always attached.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.evaluation.figures import MODEL_COLORS, MODEL_HATCH, plot_model_comparison
from src.evaluation.reporting import (
    MODEL_ORDER,
    compare_models,
    format_table,
    load_runs,
    summarize,
)


def _write_run(root, cohort, model, seed, warm, cold=None, n_warm=100, n_cold=0):
    run = root / f"{cohort}_{model}_{seed}_20260101-000000"
    run.mkdir(parents=True)
    payload = {
        "run_id": run.name, "cohort": cohort, "model": model, "seed": seed,
        "test": {
            "n_users": {"warm": n_warm, "cold": n_cold, "all": n_warm + n_cold},
            "warm": warm, "cold": cold, "all": None,
        },
    }
    (run / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    return run


@pytest.fixture
def runs_dir(tmp_path):
    """Two models x three seeds; one model varies across seeds, one does not."""
    root = tmp_path / "runs"
    root.mkdir()
    for seed in (2020, 2021, 2022):
        _write_run(root, "original", "popularity", seed,
                   warm={"recall@20": 0.01, "ndcg@20": 0.008,
                         "hit_rate@20": 0.03, "coverage@20": 0.0002})
    for seed, value in zip((2020, 2021, 2022), (0.020, 0.022, 0.024)):
        _write_run(root, "original", "bt_dkgrec", seed,
                   warm={"recall@20": value, "ndcg@20": value / 2,
                         "hit_rate@20": 0.05, "coverage@20": 0.01},
                   cold=None)
    return root


def test_load_runs_reads_every_run(runs_dir) -> None:
    frame = load_runs(runs_dir, "test", "warm")
    assert len(frame) == 6
    assert set(frame["seed"]) == {2020, 2021, 2022}


def test_load_runs_skips_unmeasured_segment_instead_of_scoring_zero(runs_dir) -> None:
    """A personalised model on cold users is unmeasured, not measured as 0."""
    cold = load_runs(runs_dir, "test", "cold")
    assert cold.empty


def test_load_runs_has_no_seed_filter() -> None:
    """Integrity rule 11: the aggregator cannot be pointed at a subset of seeds."""
    import inspect
    signature = inspect.signature(load_runs)
    assert "seed" not in signature.parameters


def test_load_runs_rejects_a_missing_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_runs(tmp_path / "khong-ton-tai")


def test_summarize_reports_mean_and_std(runs_dir) -> None:
    summary = summarize(load_runs(runs_dir, "test", "warm"), k=20)
    row = summary.loc[("original", "bt_dkgrec")]
    assert row[("recall@20", "mean")] == pytest.approx(0.022)
    assert row[("recall@20", "std")] > 0
    assert row[("n_seeds", "")] == 3


def test_summarize_gives_deterministic_model_zero_std(runs_dir) -> None:
    summary = summarize(load_runs(runs_dir, "test", "warm"), k=20)
    assert summary.loc[("original", "popularity")][("recall@20", "std")] == 0.0


def test_summarize_uses_the_canonical_model_order(runs_dir) -> None:
    summary = summarize(load_runs(runs_dir, "test", "warm"), k=20)
    models = list(summary.index.get_level_values("model"))
    assert models == [m for m in MODEL_ORDER if m in models]


def test_table_carries_the_deterministic_footnote(runs_dir) -> None:
    table = format_table(summarize(load_runs(runs_dir, "test", "warm"), k=20))
    assert "TAT DINH" in table
    assert "±" in table
    assert "BT-DKGRec-GCN" in table


def test_table_on_empty_input_says_so() -> None:
    assert "chua co run" in format_table(pd.DataFrame())


# ── Welch's t-test ──────────────────────────────────────────────────────


def test_welch_test_compares_two_models(runs_dir) -> None:
    frame = load_runs(runs_dir, "test", "warm")
    result = compare_models(frame, "bt_dkgrec", "popularity", "original", "recall@20")
    assert result["difference"] == pytest.approx(0.012)
    assert result["p_value"] is not None
    assert result["n_a"] == result["n_b"] == 3


def test_welch_test_refuses_when_both_sides_are_deterministic(runs_dir) -> None:
    frame = load_runs(runs_dir, "test", "warm")
    result = compare_models(frame, "popularity", "popularity", "original", "recall@20")
    assert result["p_value"] is None
    assert "tat dinh" in result["note"]


def test_welch_test_refuses_with_a_single_run(tmp_path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    _write_run(root, "original", "popularity", 2020,
               warm={"recall@20": 0.01, "ndcg@20": 0.01,
                     "hit_rate@20": 0.01, "coverage@20": 0.01})
    _write_run(root, "original", "bt_dkgrec", 2020,
               warm={"recall@20": 0.02, "ndcg@20": 0.02,
                     "hit_rate@20": 0.02, "coverage@20": 0.02})
    frame = load_runs(root, "test", "warm")
    result = compare_models(frame, "bt_dkgrec", "popularity", "original", "recall@20")
    assert result["p_value"] is None
    assert "2 run" in result["note"]


# ── Figures ─────────────────────────────────────────────────────────────


def test_every_model_has_a_fixed_colour_and_hatch() -> None:
    """Colour follows the entity: no model may fall back to a cycled hue."""
    assert set(MODEL_COLORS) == set(MODEL_ORDER)
    assert set(MODEL_HATCH) == set(MODEL_ORDER)
    assert len(set(MODEL_COLORS.values())) == len(MODEL_ORDER)   # no duplicates


def test_comparison_figure_is_written_in_both_formats(runs_dir, tmp_path) -> None:
    frame = load_runs(runs_dir, "test", "warm")
    png = plot_model_comparison(frame, "original", tmp_path / "figs", k=20)
    assert png.exists() and png.stat().st_size > 0
    assert png.with_suffix(".pdf").exists()
