"""Step 5+: fit a model and evaluate it end to end, writing a run artifact.

Currently serves the heuristic baselines; the graph models plug into the same
path in Buoc 6 without changing the evaluation protocol, which is the point of
the shared :class:`~src.models.base.Recommender` interface.

Produces ``experiments/runs/<cohort>_<model>_<seed>_<timestamp>/`` with
``config.yaml``, ``seed.txt``, ``env.json``, ``metrics.json``, ``topk.csv``,
``curves.csv`` va ``train.log``.

Usage::

    python scripts/03_train.py --model popularity --cohort original --seed 2020
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
from src.models.base import ModelContext  # noqa: E402
from src.models.popularity import HEURISTIC_MODELS  # noqa: E402
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

    edges_path = cfg.paths.resolved()["processed"] / cfg.cohort.name / cfg.model.name
    edges = (
        pd.read_parquet(edges_path / "edges_interacted.parquet")
        if (edges_path / "edges_interacted.parquet").exists()
        else pd.DataFrame(columns=["visitor_idx", "item_idx", "weight"])
    )
    split_info = json.loads((interim_dir / "split.json").read_text(encoding="utf-8"))

    context = ModelContext(
        cfg=cfg,
        mapping=mapping,
        train_events=train,
        interaction_edges=edges,
        t_train=int(split_info["t_train"]),
    )
    return context, events, mapping


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
    args = parser.parse_args()

    overrides: dict = {}
    if args.device:
        overrides["training"] = {"device": args.device}
    if args.eval_batch_size:
        overrides["evaluation"] = {"batch_size": args.eval_batch_size}
    overrides = overrides or None
    cfg = load_config(model=args.model, cohort=args.cohort, seed=args.seed, overrides=overrides)

    run_dir = cfg.paths.resolved()["runs"] / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    set_seed(cfg.seed)

    print("=" * 78)
    print(f"{cfg.model.name} | cohort = {cfg.cohort.name} | seed = {cfg.seed}")
    print("=" * 78)

    if cfg.model.name not in HEURISTIC_MODELS:
        raise SystemExit(
            f"model {cfg.model.name!r} chua duoc trien khai o buoc nay "
            f"(hien co: {sorted(HEURISTIC_MODELS)})"
        )

    interim_dir = cfg.paths.resolved()["interim"] / cfg.cohort.name
    context, events, mapping = build_context(cfg, interim_dir)

    model = HEURISTIC_MODELS[cfg.model.name]()
    model.fit(context)

    seen = build_seen_matrix(context.train_events, mapping)
    evaluator = Evaluator(cfg, mapping, seen)

    metrics: dict[str, object] = {
        "run_id": run_dir.name,
        "cohort": cfg.cohort.name,
        "model": cfg.model.name,
        "seed": cfg.seed,
        "model_description": model.describe(),
        "n_train_items": mapping.n_items,
        "n_train_visitors": mapping.n_visitors,
    }
    # Dung xong ground truth thi tra lai bo nho cho buoc cham diem: bang events
    # chiem hang tram MB va khong con can nua khi da co eval set.
    eval_sets = {
        split: build_evaluation_set(events, mapping, split, cfg, seen)
        for split in ("valid", "test")
    }
    del events
    gc.collect()

    rankings_by_split = {}
    for split in ("valid", "test"):
        split_metrics, rankings = evaluator.evaluate(model, eval_sets[split])
        metrics[split] = split_metrics
        rankings_by_split[split] = rankings

    # Model selection reads validation only (leakage rule 7). Heuristics have
    # nothing to select, but the curve file is still written so every run
    # carries the same evidence shape.
    valid_warm = metrics["valid"]["warm"]  # type: ignore[index]
    pd.DataFrame(
        [{
            "epoch": 0,
            "loss": None,
            "valid_" + cfg.training.monitor: valid_warm[cfg.training.monitor] if valid_warm else None,
            "note": "mo hinh khong hoc — khong co duong hoi tu",
        }]
    ).to_csv(run_dir / "curves.csv", index=False)

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
