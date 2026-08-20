"""Step 2: build the data layer for one cohort.

Reads raw RetailRocket, splits it temporally, builds train-only id mappings,
extracts admissible side information, and writes Parquet to
``data/interim/<cohort>/`` together with ``audit.json`` and ``split.json``.

The audit table is printed against the reference figures of CLAUDE.md. Those are
**sanity checks, not targets**: a new number is kept as-is and written to
``audit.json``. Only an order-of-magnitude gap means a real bug.

Usage::

    python scripts/01_preprocess.py --cohort original
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.cohort import apply_cohort, select_cohort_visitors  # noqa: E402
from src.data.loader import audit_events, load_category_tree, load_events  # noqa: E402
from src.data.mapping import IdMapping, write_node_offsets  # noqa: E402
from src.data.side_info import extract_side_info  # noqa: E402
from src.data.splitter import split_events  # noqa: E402
from src.guards.leakage import run_preprocess_guards  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("preprocess")

#: Reference figures from CLAUDE.md / thesis v11 Bang 4.2. Sanity checks only.
REFERENCE = {
    "original": {
        "train_events": 2_024_042,
        "train_visitors": 1_027_985,
        "train_items": 205_106,
        "train_target_events": 66_693,
        "valid_warm_users": 552,
        "test_warm_users": 593,
        "interaction_edges": 1_570_409,
        "n_item_category_edges": 165_528,
        "n_item_property_edges": 3_307_294,
        "n_category_parent_edges": 1_161,
        "n_graph_entities": 214_396,
    },
    "active": {
        "train_events": 707_680,
        "train_visitors": 60_559,
        "train_items": 86_659,
        "train_target_events": 48_482,
        "valid_warm_users": 234,
        "test_warm_users": 234,
        "interaction_edges": 405_128,
        "n_item_category_edges": 81_135,
        "n_item_property_edges": 1_702_617,
        "n_category_parent_edges": 1_093,
        "n_graph_entities": 131_223,
    },
}


def warm_target_users(events: pd.DataFrame, split: str, target: list[str], mapping: IdMapping) -> int:
    """Count visitors with a target event in ``split`` that also exist in train.

    Cold visitors (absent from the train mapping) have no personalised
    embedding and are never mixed into the personalised metric.
    """
    subset = events[(events["split"] == split) & events["behavior"].isin(target)]
    known = mapping.visitor_index(subset["visitorid"]) >= 0
    return int(subset.loc[known, "visitorid"].nunique())


def print_audit(title: str, rows: list[tuple[str, int, int | None]]) -> None:
    """Print one audit block with a reference column and a deviation column."""
    print(f"\n{title}")
    print(f"  {'Chi so':<34}{'Do duoc':>14}{'Tham chieu':>14}{'Lech':>12}")
    print("  " + "-" * 74)
    for name, measured, reference in rows:
        if reference is None:
            print(f"  {name:<34}{measured:>14,}{'—':>14}{'—':>12}")
            continue
        delta = measured - reference
        pct = 100 * delta / reference if reference else 0.0
        flag = "" if abs(pct) <= 5 else ("  <-- LECH >5%" if abs(pct) < 900 else "  <-- LECH 1 BAC")
        print(f"  {name:<34}{measured:>14,}{reference:>14,}{delta:>+12,}{flag}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Buoc 2: tang du lieu")
    parser.add_argument("--cohort", choices=["original", "active"], default="original")
    args = parser.parse_args()

    setup_logging()
    # Preprocessing reads only the data/cohort/graph sections, which every model
    # shares; the model name here just satisfies the config loader.
    cfg = load_config(model="bt_dkgrec", cohort=args.cohort)
    out_dir = cfg.paths.resolved()["interim"] / args.cohort
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"BT-DKGRec-GCN — tien xu ly du lieu | cohort = {args.cohort}")
    print("=" * 78)

    events = load_events(cfg)
    raw_audit = audit_events(events, cfg)

    events, boundaries = split_events(events, cfg.data.split)
    train_all = events[events["split"] == "train"]

    visitors = select_cohort_visitors(train_all, cfg.cohort)
    events = apply_cohort(events, visitors)
    train = events[events["split"] == "train"]

    mapping = IdMapping.from_train_events(train.drop(columns="split"))
    mapping.save(out_dir)

    category_tree = load_category_tree(cfg)
    side = extract_side_info(cfg, boundaries, mapping, category_tree)

    interaction_edges = int(len(train[["visitorid", "itemid"]].drop_duplicates()))
    # v11 Bang 4.2 goi dong nay la "Entity count cua graph hoc duoc". Doi chieu nguoc
    # cho thay con so 214.396 = item + category + PropertyValue, KHONG ke Visitor.
    # Bao ca hai de tranh so sanh nham dinh nghia.
    n_entities_no_visitor = mapping.n_items + len(side.categories) + len(side.property_values)
    n_nodes_total = mapping.n_visitors + n_entities_no_visitor
    write_node_offsets(
        out_dir, mapping.n_visitors, mapping.n_items,
        len(side.categories), len(side.property_values),
    )

    audit = {
        "cohort": args.cohort,
        "raw": raw_audit,
        "split": boundaries.as_dict(),
        "train_events": int(len(train)),
        "train_visitors": mapping.n_visitors,
        "train_items": mapping.n_items,
        "train_target_events": int(train["behavior"].isin(cfg.data.target_behaviors).sum()),
        "valid_events": int((events["split"] == "valid").sum()),
        "test_events": int((events["split"] == "test").sum()),
        "valid_warm_users": warm_target_users(events, "valid", cfg.data.target_behaviors, mapping),
        "test_warm_users": warm_target_users(events, "test", cfg.data.target_behaviors, mapping),
        "interaction_edges": interaction_edges,
        "n_graph_entities_excl_visitor": int(n_entities_no_visitor),
        "n_nodes_total": int(n_nodes_total),
        **side.stats,
    }
    audit["n_side_edges"] = (
        audit["n_item_category_edges"]
        + audit["n_item_property_edges"]
        + audit["n_category_parent_edges"]
    )

    # ── GATE: chay guard chong ro ri ngay tren cau truc trong bo nho ──────
    print("\nGUARD (trong bo nho)")
    passed = run_preprocess_guards(
        events=events,
        visitor_ids=mapping.visitor_ids,
        item_ids=mapping.item_ids,
        item_category=side.item_category,
        item_property=side.item_property,
        t_train=boundaries.t_train,
        t_valid_end=boundaries.t_valid_end,
        monitor=cfg.training.monitor,
    )
    for name in passed:
        print(f"  PASS  {name}")
    print("  n/a   rule 5 (candidate) va rule 6 (negative sampling) — chua co artefact")
    print("        o buoc tien xu ly; se chay o evaluator (Buoc 5) va sampler (Buoc 6)")

    events.to_parquet(out_dir / "events.parquet", index=False)
    side.item_category.to_parquet(out_dir / "side_item_category.parquet", index=False)
    side.item_property.to_parquet(out_dir / "side_item_property.parquet", index=False)
    side.property_values.to_parquet(out_dir / "side_property_values.parquet", index=False)
    side.categories.to_parquet(out_dir / "side_categories.parquet", index=False)
    side.category_parent.to_parquet(out_dir / "side_category_parent.parquet", index=False)
    (out_dir / "split.json").write_text(json.dumps(boundaries.as_dict(), indent=2), encoding="utf-8")
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # ── GATE: chay lai tren chinh file da ghi (bat loi serialize) ─────────
    print("\nGUARD (doc lai tu Parquet)")
    reloaded_mapping = IdMapping.load(out_dir)
    run_preprocess_guards(
        events=pd.read_parquet(out_dir / "events.parquet"),
        visitor_ids=reloaded_mapping.visitor_ids,
        item_ids=reloaded_mapping.item_ids,
        item_category=pd.read_parquet(out_dir / "side_item_category.parquet"),
        item_property=pd.read_parquet(out_dir / "side_item_property.parquet"),
        t_train=boundaries.t_train,
        t_valid_end=boundaries.t_valid_end,
        monitor=cfg.training.monitor,
    )
    print(f"  PASS  toan bo {len(passed)} guard tren file da ghi")

    reference = REFERENCE[args.cohort]
    print_audit(
        "A. Audit du lieu raw (khong phu thuoc cohort)",
        [
            ("events.csv tong dong", raw_audit["total_events"], 2_756_101),
            ("  view", raw_audit["n_view"], 2_664_312),
            ("  addtocart", raw_audit["n_addtocart"], 69_332),
            ("  transaction", raw_audit["n_transaction"], 22_457),
            ("target events (cart OR txn)", raw_audit["n_target_events"], 91_789),
            ("item phan biet", raw_audit["unique_items"], 235_061),
            ("visitor phan biet", raw_audit["unique_visitors"], None),
        ],
    )
    print(f"\n  T_train      = {boundaries.t_train}")
    print(f"  T_valid_end  = {boundaries.t_valid_end}")
    print(f"  span         = {boundaries.as_dict()['span_days']} ngay")

    print_audit(
        f"B. Sau split va loc cohort ({args.cohort})",
        [
            ("train events", audit["train_events"], reference["train_events"]),
            ("train visitors", audit["train_visitors"], reference["train_visitors"]),
            ("train items", audit["train_items"], reference["train_items"]),
            ("train target events", audit["train_target_events"], reference["train_target_events"]),
            ("valid warm users", audit["valid_warm_users"], reference["valid_warm_users"]),
            ("test warm users", audit["test_warm_users"], reference["test_warm_users"]),
            ("canh tuong tac sau tong hop", audit["interaction_edges"], reference["interaction_edges"]),
        ],
    )
    print_audit(
        "C. Side information",
        [
            ("item-category edges", audit["n_item_category_edges"], reference["n_item_category_edges"]),
            ("item-property edges", audit["n_item_property_edges"], reference["n_item_property_edges"]),
            ("category-parent edges", audit["n_category_parent_edges"], reference["n_category_parent_edges"]),
            ("tong side edges", audit["n_side_edges"], None),
            ("PropertyValue nodes", audit["n_property_values"], None),
            ("Category nodes", audit["n_categories"], None),
            ("entity (item+category+PV)", audit["n_graph_entities_excl_visitor"], reference["n_graph_entities"]),
            ("tong node ke ca visitor", audit["n_nodes_total"], None),
        ],
    )

    print(f"\nDa ghi: {out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.name:<34}{path.stat().st_size / 1e6:>8.2f} MB")
    print("\nLuu y: cot 'Tham chieu' la moc kiem tra hop ly, KHONG phai muc tieu phai khop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
