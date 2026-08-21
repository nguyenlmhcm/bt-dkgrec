"""Step 4: build the Behavior-Time Dynamic Knowledge Graph for one variant.

Writes ``data/processed/<cohort>/<model>/`` containing the adjacency matrix, the
aggregated interaction edges, and ``graph_stats.json``.

Three variants own a graph; they differ in exactly the two dimensions the thesis
argues about::

    lightgcn        uniform weights, no side information
    static_kg_gcn   uniform weights, with side information   <- ablation
    bt_dkgrec       behavior-time weights, with side information

``static_kg_gcn`` vs ``bt_dkgrec`` differ only in ``edge_weight()``.

Usage::

    python scripts/02_build_graph.py --cohort original --model bt_dkgrec
    python scripts/02_build_graph.py --cohort original --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.graph.builder import build_graph, save_graph  # noqa: E402
from src.graph.normalize import symmetric_normalize  # noqa: E402
from src.graph.weighting import WEIGHTING_BY_MODEL, weighting_for_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("build_graph")

GRAPH_MODELS = tuple(WEIGHTING_BY_MODEL)


def build_one(cohort: str, model: str, save_normalized: bool) -> dict[str, object]:
    """Build, validate and persist one graph variant."""
    cfg = load_config(model=model, cohort=cohort)
    paths = cfg.paths.resolved()
    interim_dir = paths["interim"] / cohort
    if not (interim_dir / "events.parquet").exists():
        raise FileNotFoundError(
            f"chua co du lieu tien xu ly cho cohort {cohort!r}: "
            f"chay `make preprocess COHORT={cohort}` truoc"
        )

    print("=" * 78)
    print(f"Dung graph | cohort = {cohort} | model = {model}")
    print("=" * 78)

    graph, edges = build_graph(cfg, interim_dir, weighting_for_model(cfg))
    out_dir = paths["processed"] / cohort / model
    save_graph(graph, edges, out_dir, save_normalized=save_normalized)

    normalized = symmetric_normalize(graph.adjacency)
    stats = dict(graph.stats)
    stats["normalized_value_min"] = float(normalized.data.min())
    stats["normalized_value_max"] = float(normalized.data.max())

    print(f"\n  node               {graph.node_space.total:>12,}")
    for name, count in graph.edge_counts.items():
        print(f"  {name:<18} {count:>12,}")
    print(f"  {'tong canh':<18} {graph.n_edges:>12,}")
    print(f"  {'phan tu khac 0':<18} {graph.adjacency.nnz:>12,}")
    print(f"  W(u,i) in         [{stats['interaction_weight_min']:.6f}, "
          f"{stats['interaction_weight_max']:.6f}]")
    print(f"  A_hat  in         [{stats['normalized_value_min']:.6g}, "
          f"{stats['normalized_value_max']:.6g}]")
    print(f"  node co lap        {stats['isolated_nodes']:>12,}")
    print(f"  -> {out_dir}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Buoc 4: dung do thi tri thuc")
    parser.add_argument("--cohort", choices=["original", "active"], default="original")
    parser.add_argument("--model", choices=GRAPH_MODELS, default="bt_dkgrec")
    parser.add_argument("--all", action="store_true", help="dung ca ba bien the co graph")
    parser.add_argument(
        "--save-normalized", action="store_true",
        help="ghi them A_hat ra dia (mac dinh tinh lai khi nap)",
    )
    args = parser.parse_args()

    setup_logging()
    models = GRAPH_MODELS if args.all else (args.model,)
    for model in models:
        build_one(args.cohort, model, args.save_normalized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
