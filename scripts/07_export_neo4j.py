#!/usr/bin/env python
"""Buoc 10 — xuat lop truy vet ra CSV cho Neo4j.

Chay:
    make neo4j COHORT=original
    python scripts/07_export_neo4j.py --cohort original --model bt_dkgrec

Mac dinh chi xuat visitor CO trong tap danh gia (`--scope evaluated`). Do thi
day du cua cohort Original co 1,03 trieu visitor va 4,66 trieu canh — nap het
vao Neo4j chi de demo mot visitor la khong can thiet. `--scope all` van co san
khi can dung lai toan bo do thi.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.export.neo4j_export import export_csv  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Buoc 10 — xuat Neo4j")
    parser.add_argument("--cohort", default="original")
    parser.add_argument("--model", default="bt_dkgrec")
    parser.add_argument("--scope", default="evaluated", choices=("evaluated", "all"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(model=args.model, cohort=args.cohort)
    paths = cfg.paths.resolved()
    interim_dir = paths["interim"] / args.cohort
    graph_dir = paths["processed"] / args.cohort / args.model
    out_dir = args.out or Path("data/neo4j") / f"{args.cohort}_{args.model}_{args.scope}"

    for required in (interim_dir / "events.parquet", graph_dir / "edges_interacted.parquet"):
        if not required.exists():
            raise SystemExit(
                f"LOI: thieu {required}. Chay `make preprocess` va `make graph` truoc."
            )

    log.info("xuat %s/%s (scope=%s) -> %s", args.cohort, args.model, args.scope, out_dir)
    manifest = export_csv(cfg, interim_dir, graph_dir, out_dir, scope=args.scope)

    print(f"\nDa xuat vao {out_dir}/")
    for name, n in manifest["counts"].items():
        print(f"  {name:<20}{n:>12,}")
    check = manifest["layer_consistency"]
    print(f"\nHai lop khop: {check['n_pairs']:,} cap, {check['n_events']:,} su kien, "
          f"lech toi da {check['worst_relative_difference']:.3e}")
    print(f"\nNap vao Neo4j: chep *.csv vao thu muc import/ roi chay import.cypher")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
