"""Typed configuration loading and validation.

Configs are layered and merged in a fixed order::

    configs/base.yaml
      <- configs/data/<cohort>.yaml
      <- configs/models/<model>.yaml
      <- explicit CLI overrides

Every section is a pydantic v2 model with ``extra="forbid"``, so a typo in a
YAML key raises instead of being silently ignored. No hyperparameter is ever
hardcoded in Python; the resolved config is snapshotted into each run artifact
as ``config.yaml`` so a run can be reproduced from its own directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
SEEDS_FILE = REPO_ROOT / "experiments" / "seeds.json"

COHORTS = ("original", "active")
MODELS = ("popularity", "recent_popularity", "lightgcn", "static_kg_gcn", "bt_dkgrec")


class _Strict(BaseModel):
    """Base for every config section: unknown keys are errors, values immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(_Strict):
    name: str
    thesis: str


class PathsConfig(_Strict):
    raw: Path
    interim: Path
    processed: Path
    runs: Path

    def resolved(self, root: Path = REPO_ROOT) -> dict[str, Path]:
        """Return the paths anchored at the repository root."""
        return {k: (root / v) for k, v in self.model_dump().items()}


class SplitConfig(_Strict):
    """Temporal split.

    ``mode="time_span"`` cuts on quantiles of the *time range*, not on event
    counts. Verified against the v11 thesis (Bang 4.2): with
    ``T_train = t_min + int(0.7 * (t_max - t_min))`` and an inclusive boundary,
    RetailRocket yields exactly 2,024,042 train events.
    """

    mode: Literal["time_span"]
    train_ratio: float = Field(gt=0.0, lt=1.0)
    valid_ratio: float = Field(gt=0.0, lt=1.0)
    test_ratio: float = Field(gt=0.0, lt=1.0)
    boundary_inclusive: bool

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> "SplitConfig":
        total = self.train_ratio + self.valid_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        return self


class DataConfig(_Strict):
    events_file: str
    item_properties_files: list[str]
    category_tree_file: str
    chunksize: int = Field(gt=0)
    history_behaviors: list[str]
    target_behaviors: list[str]
    split: SplitConfig

    @model_validator(mode="after")
    def _targets_subset_of_history(self) -> "DataConfig":
        missing = set(self.target_behaviors) - set(self.history_behaviors)
        if missing:
            raise ValueError(f"target_behaviors not in history_behaviors: {sorted(missing)}")
        return self


class CohortConfig(_Strict):
    name: Literal["original", "active"]
    min_active_events: int = Field(ge=0)

    @model_validator(mode="after")
    def _threshold_matches_name(self) -> "CohortConfig":
        if self.name == "original" and self.min_active_events != 0:
            raise ValueError("cohort 'original' must not filter users (min_active_events=0)")
        if self.name == "active" and self.min_active_events < 1:
            raise ValueError("cohort 'active' requires min_active_events >= 1")
        return self


class GraphConfig(_Strict):
    min_pv_freq: int = Field(ge=1)
    max_property_nodes: int = Field(gt=0)
    drop_properties: list[str]
    rel_weight_item_category: float = Field(gt=0.0)
    rel_weight_item_property: float = Field(gt=0.0)
    rel_weight_category_parent: float = Field(gt=0.0)


class WeightingConfig(_Strict):
    """Parameters of formulas (3.16)-(3.18).

    These are *parameters only*. Which model applies the formula is decided by
    the model class overriding ``edge_weight()`` -- never by a config switch.
    """

    alpha: dict[str, float]
    lambda_decay: float = Field(ge=0.0)
    d_day: int = Field(gt=0)

    @field_validator("alpha")
    @classmethod
    def _alpha_positive(cls, v: dict[str, float]) -> dict[str, float]:
        bad = {k: x for k, x in v.items() if x <= 0}
        if bad:
            raise ValueError(f"behavior weights must be > 0, got {bad}")
        return v


class ModelConfig(_Strict):
    name: Literal["popularity", "recent_popularity", "lightgcn", "static_kg_gcn", "bt_dkgrec"]
    kind: Literal["heuristic", "gcn"]
    use_side_info: bool
    trainable: bool
    embedding_dim: int | None = Field(default=None, gt=0)
    num_layers: int | None = Field(default=None, ge=1)
    recent_window_days: int | None = Field(default=None, gt=0)
    popularity_signal: Literal["target", "all"] | None = None

    @model_validator(mode="after")
    def _gcn_needs_dims(self) -> "ModelConfig":
        if self.kind == "gcn" and (self.embedding_dim is None or self.num_layers is None):
            raise ValueError(f"model '{self.name}' is a GCN and needs embedding_dim and num_layers")
        if self.kind == "heuristic" and self.trainable:
            raise ValueError(f"model '{self.name}' is heuristic and cannot be trainable")
        return self


class TrainingConfig(_Strict):
    loss: Literal["bpr", "weighted_bpr"]
    num_negatives: int = Field(ge=1)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    reg_weight: float = Field(ge=0.0)
    max_epochs: int = Field(ge=300)
    eval_every: int = Field(ge=1)
    patience: int = Field(ge=1)
    monitor: str
    monitor_mode: Literal["max", "min"]
    device: str

    @field_validator("monitor")
    @classmethod
    def _monitor_is_valid_metric(cls, v: str) -> str:
        if "test" in v.lower():
            raise ValueError("model selection must never read a test metric (leakage rule 7)")
        return v


class EvaluationConfig(_Strict):
    batch_size: int = Field(gt=0)
    k_values: list[int]
    primary_k: int
    filter_seen: bool
    candidate_scope: Literal["train_items"]
    coverage_denominator: Literal["train_items"]
    segments: list[str]

    @model_validator(mode="after")
    def _primary_k_in_k_values(self) -> "EvaluationConfig":
        if self.primary_k not in self.k_values:
            raise ValueError(f"primary_k={self.primary_k} must be one of k_values={self.k_values}")
        return self


class Config(_Strict):
    """Fully resolved, validated experiment configuration."""

    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    cohort: CohortConfig
    graph: GraphConfig
    weighting: WeightingConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    seed: int

    @model_validator(mode="after")
    def _weighted_bpr_is_ablation_only(self) -> "Config":
        # Decision C: the whole 5-model x 3-seed x 2-cohort matrix uses standard
        # BPR (3.30). Weighted BPR (3.31) is a single ablation row of bt_dkgrec.
        if self.training.loss == "weighted_bpr" and self.model.name != "bt_dkgrec":
            raise ValueError(
                "weighted_bpr is an ablation of bt_dkgrec only; the main matrix uses standard bpr"
            )
        return self

    @property
    def run_id(self) -> str:
        """Run artifact directory name: ``<cohort>_<model>_<seed>_<timestamp>``."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{self.cohort.name}_{self.model.name}_{self.seed}_{stamp}"

    def to_yaml(self) -> str:
        """Serialise for the ``config.yaml`` snapshot inside a run artifact."""
        return yaml.safe_dump(json.loads(self.model_dump_json()), sort_keys=False, allow_unicode=True)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_seeds(path: Path = SEEDS_FILE) -> list[int]:
    """Return the official seed list. All models run on exactly these seeds."""
    with path.open(encoding="utf-8") as fh:
        return list(json.load(fh)["seeds"])


def load_config(
    model: str,
    cohort: str = "original",
    seed: int | None = None,
    overrides: dict[str, Any] | None = None,
    config_dir: Path = CONFIG_DIR,
) -> Config:
    """Load, merge and validate the configuration for one experiment run.

    Args:
        model: One of ``MODELS``.
        cohort: ``"original"`` (main evaluation set) or ``"active"`` (exploratory).
        seed: Overrides ``base.yaml``'s seed when given.
        overrides: Nested dict merged last, e.g. ``{"training": {"device": "cpu"}}``.
        config_dir: Root of the YAML config tree (injectable for tests).

    Returns:
        A validated, frozen :class:`Config`.

    Raises:
        ValueError: Unknown model/cohort name.
        pydantic.ValidationError: Any invalid or unknown config key.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort {cohort!r}; expected one of {COHORTS}")

    merged = _read_yaml(config_dir / "base.yaml")
    merged = _deep_merge(merged, _read_yaml(config_dir / "data" / f"{cohort}.yaml"))
    merged = _deep_merge(merged, _read_yaml(config_dir / "models" / f"{model}.yaml"))
    if seed is not None:
        merged = _deep_merge(merged, {"seed": seed})
    if overrides:
        merged = _deep_merge(merged, overrides)
    return Config.model_validate(merged)
