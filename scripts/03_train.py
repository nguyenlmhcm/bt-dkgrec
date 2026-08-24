"""Step 5+: fit a model and evaluate it end to end, writing a run artifact.

Serves every model in the matrix through one path: the heuristic baselines of
Buoc 5 and the graph models of Buoc 6-8. Nothing about the evaluation protocol
depends on which one is running -- same candidate set, same seen-filtering, same
metrics -- which is what makes the comparison admissible.

The one structural difference is *when* the validation set is built. A trainable
model needs it **before** ``fit()``, because early stopping reads the validation
metric while training. It is still only the validation split: leakage rule 7 is
asserted inside the trainer before the first gradient step.

Produces ``experiments/runs/<cohort>_<model>_<seed>_<timestamp>/`` with
``config.yaml``, ``seed.txt``, ``env.json``, ``metrics.json``, ``topk.csv``,
``curves.csv`` va ``train.log``.

Usage::

    python scripts/03_train.py --model popularity --cohort original --seed 2020
    python scripts/03_train.py --model bt_dkgrec --cohort active --seed 2020
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.mapping import IdMapping  # noqa: E402
from src.evaluation.evaluator import Evaluator, build_evaluation_set, build_seen_matrix  # noqa: E402
from src.models.base import ModelContext, Recommender  # noqa: E402
from src.models.registry import build_model, is_trainable  # noqa: E402
from src.training.seeding import set_seed  # noqa: E402
from src.utils.config import Config, load_config  # noqa: E402
from src.utils.environment import environment_record  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("train")


def build_context(cfg: Config, interim_dir: Path) -> tuple[ModelContext, pd.DataFrame, IdMapping]:
    """Load train material and attach matrix indices."""
    events = pd.read_parquet(interim_dir / "events.parquet")
    mapping = IdMapping.load(interim_dir)
    events["visitor_idx"] = mapping.visitor_index(events["visitorid"])
    events["item_idx"] = mapping.item_index(events["itemid"])
    # Copy so the parent frame can be released later: a pandas slice keeps the
    # whole original alive, which defeats the point of freeing it.
    train = events[events["split"] == "train"][
        ["visitor_idx", "item_idx", "behavior", "timestamp"]
    ].copy()

    graph_dir = cfg.paths.resolved()["processed"] / cfg.cohort.name / cfg.model.name
    edges = (
        pd.read_parquet(graph_dir / "edges_interacted.parquet")
        if (graph_dir / "edges_interacted.parquet").exists()
        else pd.DataFrame(columns=["visitor_idx", "item_idx", "weight"])
    )
    split_info = json.loads((interim_dir / "split.json").read_text(encoding="utf-8"))

    context = ModelContext(
        cfg=cfg,
        mapping=mapping,
        train_events=train,
        interaction_edges=edges,
        t_train=int(split_info["t_train"]),
        graph_dir=graph_dir if graph_dir.exists() else None,
    )
    return context, events, mapping


def heuristic_curve(cfg: Config, valid_metrics: dict | None) -> pd.DataFrame:
    """A one-row ``curves.csv`` for models that do not learn.

    Written anyway so every run carries the same evidence shape and Buoc 9 can
    read one schema instead of two.
    """
    warm = (valid_metrics or {}).get("warm")
    return pd.DataFrame(
        [{
            "epoch": 0,
            "loss": None,
            f"valid_{cfg.training.monitor}": warm[cfg.training.monitor] if warm else None,
            "note": "mo hinh khong hoc — khong co duong hoi tu",
        }]
    )


def print_training_summary(model: Recommender) -> None:
    """Show convergence facts on stdout so a Colab log carries them too."""
    result = getattr(model, "training_result", None)
    if result is None:
        return
    facts = result.describe()
    print("\nHUAN LUYEN")
    print(f"  epoch da chay      {facts['n_epochs']}")
    print(f"  epoch tot nhat     {facts['best_epoch']}  (valid {facts['monitor']} = "
          f"{facts['best_valid_metric']})")
    print(f"  dung som           {'co' if facts['stopped_early'] else 'khong'}")
    print(f"  loss dau -> cuoi   {facts['first_loss']:.6f} -> {facts['last_loss']:.6f}")
    print(f"  thoi gian          {facts['seconds']}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Huan luyen + danh gia mot mo hinh")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cohort", choices=["original", "active"], default="original")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--eval-batch-size", type=int, default=None,
        help="ghi de evaluation.batch_size (tham so ha tang, khong doi ket qua)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help="ghi de training.max_epochs (san >= 300 do config chan — xem CLAUDE.md); "
             "dung cho smoke test, ket qua bao cao chay theo configs/",
    )
    args = parser.parse_args()

    overrides: dict = {}
    if args.device:
        overrides["training"] = {"device": args.device}
    if args.eval_batch_size:
        overrides["evaluation"] = {"batch_size": args.eval_batch_size}
    if args.max_epochs:
        overrides.setdefault("training", {})["max_epochs"] = args.max_epochs
    overrides = overrides or None
    cfg = load_config(model=args.model, cohort=args.cohort, seed=args.seed, overrides=overrides)

    run_dir = cfg.paths.resolved()["runs"] / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    set_seed(cfg.seed)

    print("=" * 78)
    print(f"{cfg.model.name} | cohort = {cfg.cohort.name} | seed = {cfg.seed}")
    print("=" * 78)

    interim_dir = cfg.paths.resolved()["interim"] / cfg.cohort.name
    context, events, mapping = build_context(cfg, interim_dir)
    model = build_model(cfg)

    seen = build_seen_matrix(context.train_events, mapping)
    evaluator = Evaluator(cfg, mapping, seen)

    # Ground truth is built before fitting because a trainable model needs the
    # validation set during training. Dung xong thi tra lai bo nho: bang events
    # chiem hang tram MB va khong con can nua khi da co eval set.
    eval_sets = {
        split: build_evaluation_set(events, mapping, split, cfg, seen)
        for split in ("valid", "test")
    }
    del events
    gc.collect()

    if is_trainable(cfg):
        def validate(fitted: Recommender) -> float | None:
            """Monitored metric on VALID only — never test (leakage rule 7)."""
            split_metrics, _ = evaluator.evaluate(fitted, eval_sets["valid"])
            warm = split_metrics["warm"]
            return warm[cfg.training.monitor] if warm else None

        model.attach_validation(validate)  # type: ignore[attr-defined]

    model.fit(context)

    metrics: dict[str, object] = {
        "run_id": run_dir.name,
        "cohort": cfg.cohort.name,
        "model": cfg.model.name,
        "seed": cfg.seed,
        "model_description": model.describe(),
        "n_train_items": mapping.n_items,
        "n_train_visitors": mapping.n_visitors,
    }

    rankings_by_split = {}
    for split in ("valid", "test"):
        split_metrics, rankings = evaluator.evaluate(model, eval_sets[split])
        metrics[split] = split_metrics
        rankings_by_split[split] = rankings

    result = getattr(model, "training_result", None)
    curves = result.curves if result is not None else heuristic_curve(cfg, metrics["valid"])  # type: ignore[arg-type]
    curves.to_csv(run_dir / "curves.csv", index=False)

    evaluator.top_k_frame(eval_sets["test"], rankings_by_split["test"], model).to_csv(
        run_dir / "topk.csv", index=False
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "seed.txt").write_text(f"{cfg.seed}\n", encoding="utf-8")
    # Colab khong ghim phien ban thu vien (D25), nen ban THUC TE phai di
    # cung con so — neu khong, khong ai tai dung duoc run nay.
    (run_dir / "env.json").write_text(
        json.dumps(environment_record(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")

    print_training_summary(model)
    for split in ("valid", "test"):
        block = metrics[split]  # type: ignore[index]
        warm = block["warm"]  # type: ignore[index]
        counts = block["n_users"]  # type: ignore[index]
        print(f"\n{split.upper()}  (warm={counts['warm']}, cold={counts['cold']})")
        if warm:
            for key in sorted(warm):
                print(f"  warm  {key:<14}{warm[key]:.6f}")
    print(f"\n-> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
