"""Tests for the layered config loader.

These lock in the rules from CLAUDE.md that must never silently regress:
unknown keys fail, the experiment matrix validates, and the leakage/fairness
guarantees encoded in the schema hold.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.utils.config import CONFIG_DIR, COHORTS, MODELS, Config, load_config, load_seeds


def test_official_seeds_are_exactly_three() -> None:
    assert load_seeds() == [2020, 2021, 2022]


@pytest.mark.parametrize("cohort", COHORTS)
@pytest.mark.parametrize("model", MODELS)
def test_every_matrix_cell_validates(model: str, cohort: str) -> None:
    cfg = load_config(model=model, cohort=cohort, seed=2020)
    assert cfg.model.name == model
    assert cfg.cohort.name == cohort
    assert cfg.seed == 2020


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        load_config(model="bt_dkgrec", overrides={"training": {"lerning_rate": 0.1}})


def test_unknown_model_and_cohort_are_rejected() -> None:
    with pytest.raises(ValueError):
        load_config(model="mbht")
    with pytest.raises(ValueError):
        load_config(model="bt_dkgrec", cohort="random")


def test_split_is_temporal_and_ratios_sum_to_one() -> None:
    split = load_config(model="popularity").data.split
    assert split.mode == "time_span"
    assert split.train_ratio + split.valid_ratio + split.test_ratio == pytest.approx(1.0)


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        load_config(model="popularity", overrides={"data": {"split": {"train_ratio": 0.8}}})


def test_model_selection_cannot_monitor_a_test_metric() -> None:
    """Leakage rule 7: model selection reads valid metrics only."""
    with pytest.raises(ValidationError):
        load_config(model="bt_dkgrec", overrides={"training": {"monitor": "test_ndcg@20"}})


def test_weighted_bpr_is_restricted_to_bt_dkgrec_ablation() -> None:
    """Decision C: the main matrix uses standard BPR for every model."""
    cfg = load_config(model="bt_dkgrec", overrides={"training": {"loss": "weighted_bpr"}})
    assert cfg.training.loss == "weighted_bpr"
    with pytest.raises(ValidationError):
        load_config(model="static_kg_gcn", overrides={"training": {"loss": "weighted_bpr"}})


def test_main_matrix_defaults_to_standard_bpr() -> None:
    for model in MODELS:
        assert load_config(model=model).training.loss == "bpr"


def test_all_models_share_one_training_budget() -> None:
    """Fairness rule 9: no baseline may be given a smaller budget."""
    budgets = {
        load_config(model=m).training.max_epochs for m in MODELS
    }
    assert budgets == {1000}


def test_early_stopping_budget_is_uniform_within_a_cohort() -> None:
    """The budget that actually binds is ``patience``, not ``max_epochs``.

    Runs stop on patience long before epoch 1000, so a per-model patience would
    be a per-model budget wearing a different name. Raising it for the proposed
    model alone is the exact failure Shehzad & Jannach (RecSys '23) show makes
    any method look superior. It is therefore a COHORT-level decision, set in
    ``configs/data/<cohort>.yaml`` and never in ``configs/models/*``.
    """
    for cohort in COHORTS:
        budgets = {load_config(model=m, cohort=cohort).training.patience for m in MODELS}
        assert len(budgets) == 1, f"{cohort}: patience khong dong deu giua cac mo hinh — {budgets}"


def test_no_model_config_may_set_its_own_stopping_budget() -> None:
    """Enforced at the source: the key must not appear in any model YAML."""
    import yaml

    for path in sorted((CONFIG_DIR / "models").glob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        training = loaded.get("training") or {}
        for key in ("patience", "max_epochs", "eval_every"):
            assert key not in training, f"{path.name} tu dat {key} — ngan sach phai o cap cohort"


def test_cohort_threshold_matches_cohort_name() -> None:
    assert load_config(model="popularity", cohort="original").cohort.min_active_events == 0
    assert load_config(model="popularity", cohort="active").cohort.min_active_events == 5
    with pytest.raises(ValidationError):
        load_config(model="popularity", cohort="original", overrides={"cohort": {"min_active_events": 5}})


def test_ablation_pair_differs_only_in_model_identity() -> None:
    """static_kg_gcn vs bt_dkgrec: same side info, same budget, same everything else.

    The single difference is edge_weight(), which lives in the model class --
    not in config. So their configs must agree on every shared section.
    """
    static = load_config(model="static_kg_gcn", cohort="active", seed=2021)
    dynamic = load_config(model="bt_dkgrec", cohort="active", seed=2021)

    assert static.model.use_side_info == dynamic.model.use_side_info is True
    assert static.model.embedding_dim == dynamic.model.embedding_dim
    assert static.model.num_layers == dynamic.model.num_layers
    for section in ("data", "cohort", "graph", "weighting", "training", "evaluation"):
        assert getattr(static, section) == getattr(dynamic, section), section


def test_lightgcn_uses_no_side_information() -> None:
    assert load_config(model="lightgcn").model.use_side_info is False


def test_heuristic_models_are_not_trainable() -> None:
    for model in ("popularity", "recent_popularity"):
        cfg = load_config(model=model)
        assert cfg.model.kind == "heuristic"
        assert cfg.model.trainable is False


def test_coverage_and_candidate_scope_are_train_only() -> None:
    """Leakage rule 5 and the agreed Coverage@K denominator."""
    ev = load_config(model="bt_dkgrec").evaluation
    assert ev.candidate_scope == "train_items"
    assert ev.coverage_denominator == "train_items"
    assert ev.filter_seen is True
    assert ev.primary_k == 20


def test_behavior_weights_follow_the_intent_ordering() -> None:
    alpha = load_config(model="bt_dkgrec").weighting.alpha
    assert alpha["view"] < alpha["addtocart"] < alpha["transaction"]


def test_config_is_frozen_and_snapshottable() -> None:
    cfg = load_config(model="bt_dkgrec")
    with pytest.raises(ValidationError):
        cfg.seed = 999  # type: ignore[misc]
    assert "bt_dkgrec" in cfg.to_yaml()
    assert cfg.run_id.startswith("original_bt_dkgrec_2020_")


def test_run_id_encodes_cohort_model_seed() -> None:
    cfg: Config = load_config(model="lightgcn", cohort="active", seed=2022)
    assert cfg.run_id.startswith("active_lightgcn_2022_")
