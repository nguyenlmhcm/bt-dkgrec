"""Tests for the shape and internal agreement of a run artifact (Buoc 6-9).

These are torch-free on purpose: they guard the *evidence*, not the model, so
they must stay runnable on the VPS where torch is deliberately absent (D28).

Two invariants are defended here:

* every run writes ``curves.csv`` with the same columns, whether the model
  learned or not -- otherwise Buoc 9 has to special-case its own inputs;
* the metric a run reports belongs to the parameters the run actually kept.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.guards.consistency import (
    FATAL_RELATIVE_GAP,
    ConsistencyError,
    assert_selection_restored,
)
from src.training.curves import (
    FIXED_HEAD,
    FIXED_TAIL,
    IDENTITY_COLUMNS,
    add_identity,
    order_columns,
    valid_column,
)
from src.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _train_script():
    """Import ``scripts/03_train.py`` as a module.

    Loaded by path because ``scripts/`` is not a package. Importing it must stay
    torch-free -- the heuristic baselines run on the VPS through this very
    script -- so this import doubles as a regression test for that.
    """
    spec = importlib.util.spec_from_file_location(
        "train_script", REPO_ROOT / "scripts" / "03_train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ──────────────────────────────────────────────────────────────────────────
# curves.csv schema
# ──────────────────────────────────────────────────────────────────────────


def test_order_columns_produces_the_canonical_order() -> None:
    frame = pd.DataFrame(
        [{"note": "best", "valid_recall@20": 0.1, "epoch": 5, "loss": 0.9,
          "valid_ndcg@20": 0.2, "seconds": 1.0, "evaluated": True}]
    )
    ordered = order_columns(frame, monitor="ndcg@20")

    assert list(ordered.columns) == [
        *FIXED_HEAD, "valid_ndcg@20", "valid_recall@20", *FIXED_TAIL
    ]


def test_order_columns_creates_the_monitor_column_when_nothing_validated() -> None:
    """A run without validation must still concatenate with runs that had it."""
    frame = pd.DataFrame([{"epoch": 1, "loss": 0.9, "seconds": 0.2, "evaluated": False, "note": ""}])
    ordered = order_columns(frame, monitor="ndcg@20")

    assert valid_column("ndcg@20") in ordered.columns
    assert ordered["valid_ndcg@20"].isna().all()


def test_add_identity_puts_the_join_keys_first() -> None:
    """A run id cannot be split back into cohort and model by string surgery:
    model names contain underscores, so ``recent_popularity`` is ambiguous."""
    frame = pd.DataFrame([{"epoch": 1, "loss": 0.5}])
    stamped = add_identity(frame, model="recent_popularity", cohort="original", seed=2020)

    assert list(stamped.columns)[: len(IDENTITY_COLUMNS)] == list(IDENTITY_COLUMNS)
    assert stamped["model"].iloc[0] == "recent_popularity"
    assert stamped["cohort"].iloc[0] == "original"
    assert stamped["seed"].iloc[0] == 2020


def test_a_model_that_does_not_learn_writes_the_same_columns_as_one_that_does() -> None:
    """★ One schema for Buoc 9, whatever produced the run."""
    module = _train_script()
    cfg = load_config(model="popularity", cohort="original", seed=2020)
    warm = {"recall@20": 0.03, "ndcg@20": 0.02, "hit_rate@20": 0.04, "coverage@20": 0.001}

    heuristic = module.heuristic_curve(cfg, {"warm": warm})
    # The same block as a trained model would emit on an evaluated epoch.
    trained = order_columns(
        pd.DataFrame(
            [{"epoch": 5, "loss": 0.9, "seconds": 1.0, "evaluated": True, "note": "best",
              **{valid_column(name): score for name, score in warm.items()}}]
        ),
        monitor=cfg.training.monitor,
    )

    assert list(heuristic.columns) == list(trained.columns)
    assert len(heuristic) == 1
    assert heuristic["evaluated"].iloc[0] is True or bool(heuristic["evaluated"].iloc[0])
    assert heuristic["valid_ndcg@20"].iloc[0] == pytest.approx(0.02)


def test_heuristic_curve_survives_a_segment_with_no_measured_metric() -> None:
    """A cold-only split yields no warm block; the file is still well formed."""
    module = _train_script()
    cfg = load_config(model="popularity", cohort="original", seed=2020)

    curve = module.heuristic_curve(cfg, {"warm": None})

    assert list(curve.columns)[: len(FIXED_HEAD)] == list(FIXED_HEAD)
    assert curve["valid_ndcg@20"].isna().all()


# ──────────────────────────────────────────────────────────────────────────
# The reported model is the selected model
# ──────────────────────────────────────────────────────────────────────────


def test_identical_numbers_pass_and_are_recorded() -> None:
    record = assert_selection_restored("ndcg@20", best_epoch=45, best_value=0.0213, reevaluated=0.0213)

    assert record["best_epoch"] == 45
    assert record["relative_difference"] == pytest.approx(0.0)
    assert record["value_after_restore"] == pytest.approx(0.0213)


def test_float_noise_is_tolerated() -> None:
    """Sparse CUDA kernels are not bit-exact; the guard must not cry wolf."""
    best = 0.0213
    record = assert_selection_restored(
        "ndcg@20", best_epoch=45, best_value=best, reevaluated=best * (1 + 1e-9)
    )

    assert record["relative_difference"] < FATAL_RELATIVE_GAP


def test_a_restore_that_did_not_take_effect_is_caught() -> None:
    """★ The regression test for the bug this guard exists for.

    A trainer that reports the best epoch but leaves the *last* (or the random
    initial) parameters in place produces exactly this: a good number recorded
    during training, a much worse one when the same split is scored again.
    """
    with pytest.raises(ConsistencyError, match="KHONG phai tham so ma validation da chon"):
        assert_selection_restored("ndcg@20", best_epoch=45, best_value=0.0213, reevaluated=0.0011)


def test_a_split_that_can_no_longer_be_scored_is_caught() -> None:
    with pytest.raises(ConsistencyError, match="khong duoc nap lai"):
        assert_selection_restored("ndcg@20", best_epoch=45, best_value=0.0213, reevaluated=None)


def test_the_threshold_is_the_boundary_it_claims_to_be() -> None:
    """Just inside passes, just outside raises -- no silent band in between."""
    best = 0.02
    inside = best * (1 + FATAL_RELATIVE_GAP * 0.9)
    outside = best * (1 + FATAL_RELATIVE_GAP * 1.1)

    assert assert_selection_restored("ndcg@20", 45, best, inside)["relative_difference"] < FATAL_RELATIVE_GAP
    with pytest.raises(ConsistencyError):
        assert_selection_restored("ndcg@20", 45, best, outside)


def test_a_zero_valued_metric_does_not_divide_by_zero() -> None:
    """A model that scored 0.0 on valid is degenerate, not a crash."""
    record = assert_selection_restored("ndcg@20", best_epoch=5, best_value=0.0, reevaluated=0.0)
    assert np.isfinite(record["relative_difference"])
