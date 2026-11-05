"""Illustrative shop catalogue built on the real RetailRocket structure.

Real: item id, category tree, price property 790 (encoded, read as VND), availability,
and popularity counts. Invented: product names and customer names, see `names.py`.
Only records with ``timestamp <= T_train`` are read, like every other side input.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.shop.names import COLOURS, DEPARTMENTS, NO_COLOUR, PERSONAS, UNCATEGORISED

PRICE_FLOOR, PRICE_CAP, PRICE_STEP = 15_000, 9_000_000, 1_000
COLUMNS = ["item_id", "name", "variant", "tone", "department", "dept_idx", "icon", "product_type", "category_id",
           "price", "price_is_real", "available", "n_view", "n_cart", "n_txn"]


def stable_hash(*parts: object) -> int:
    """Hash that does not change between runs, unlike the builtin ``hash``."""
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def round_price(value: float) -> int:
    """Read an encoded price as VND, rounded the way a shop would show it."""
    return min(PRICE_CAP, max(PRICE_FLOOR, int(round(value / PRICE_STEP)) * PRICE_STEP))


#: Fixed names for the customers shown on the sign-in screen, so none repeats there.
SHOWCASE = {437888: "Thu Hà", 962057: "Minh Anh", 481437: "Quốc Bảo", 282844: "Gia Huy", 468111: "Ngọc Lan", 553749: "Hoàng Nam"}


def persona_name(visitor_id: int) -> str:
    if visitor_id in SHOWCASE:
        return SHOWCASE[visitor_id]
    return PERSONAS[stable_hash("persona", visitor_id) % len(PERSONAS)]


def describe_item(item_id: int, dept_idx: int | None, category_id: int | None) -> dict:
    """Name one item: its category decides the type, its id only the variant."""
    tone = stable_hash("tone", item_id) % len(COLOURS)
    if dept_idx is None:
        return {"name": f"Sản phẩm {item_id}", "variant": f"mã {item_id}", "tone": tone, "department": UNCATEGORISED,
                "dept_idx": -1, "icon": "package", "product_type": UNCATEGORISED}
    department, icon, types, variants = DEPARTMENTS[dept_idx]
    product_type = types[stable_hash("type", category_id) % len(types)]
    plain = [v for v in variants if not v.startswith("màu")]
    variant = plain[stable_hash("variant", item_id) % len(plain)]
    if department not in NO_COLOUR and "màu" not in product_type:
        variant = f"{variant}, màu {COLOURS[tone]}"
    return {"name": f"{product_type} {variant}", "variant": variant, "tone": tone, "department": department,
            "dept_idx": dept_idx, "icon": icon, "product_type": product_type}


def latest_properties(raw_files: list[Path], t_train: int, keep_items: set[int]) -> pd.DataFrame:
    """Latest price and availability per item with ``timestamp <= t_train``."""
    rows = []
    for path in raw_files:
        for chunk in pd.read_csv(path, chunksize=2_000_000, dtype={"property": "string", "value": "string"}):
            chunk = chunk[chunk["property"].isin(["790", "available"]) & (chunk["timestamp"] <= t_train)]
            rows.append(chunk[chunk["itemid"].isin(keep_items)])
    props = pd.concat(rows, ignore_index=True).sort_values("timestamp")
    last = props.drop_duplicates(["itemid", "property"], keep="last")
    table = last.pivot(index="itemid", columns="property", values="value")
    out = pd.DataFrame(index=table.index)
    out["raw_price"] = pd.to_numeric(table.get("790", pd.Series(dtype="string")).str.lstrip("n"), errors="coerce")
    out["raw_available"] = table.get("available", pd.Series(dtype="string"))
    return out


def build_catalog(interim: Path, raw: Path) -> pd.DataFrame:
    """Build the catalogue for one cohort's items."""
    split = json.loads((interim / "split.json").read_text(encoding="utf-8"))
    t_train = int(split["t_train"])
    items = pd.read_parquet(interim / "items.parquet")
    item_category = pd.read_parquet(interim / "side_item_category.parquet")
    parents = pd.read_parquet(interim / "side_category_parent.parquet")
    parent = dict(zip(parents["category_idx"], parents["parent_idx"]))

    def root_of(idx: int) -> int:
        for _ in range(16):
            if idx not in parent:
                break
            idx = parent[idx]
        return idx

    item_category["root"] = item_category["category_idx"].map(root_of)
    ranked_roots = item_category["root"].value_counts().index.tolist()
    if len(ranked_roots) > len(DEPARTMENTS):
        raise ValueError(f"{len(ranked_roots)} root categories, {len(DEPARTMENTS)} departments")
    dept_of_root = {root: i for i, root in enumerate(ranked_roots)}
    by_item = item_category.set_index("item_idx")

    props = latest_properties(sorted(raw.glob("item_properties_part*.csv")), t_train, set(items["item_id"]))

    events = pd.read_parquet(interim / "events.parquet", columns=["itemid", "behavior", "split"])
    counts = (events[events["split"] == "train"].groupby(["itemid", "behavior"], observed=True).size()
              .unstack(fill_value=0).reindex(columns=["view", "addtocart", "transaction"], fill_value=0))

    rows = []
    for item_idx, item_id in zip(items["idx"], items["item_id"]):
        item_id = int(item_id)
        known = item_idx in by_item.index
        category_id = int(by_item.at[item_idx, "category_id"]) if known else None
        dept_idx = dept_of_root[by_item.at[item_idx, "root"]] if known else None
        row = describe_item(item_id, dept_idx, category_id)
        raw_price = props.at[item_id, "raw_price"] if item_id in props.index else np.nan
        is_real = bool(pd.notna(raw_price) and raw_price > 0)
        price = round_price(raw_price) if is_real else round_price(40_000 + stable_hash("price", item_id) % 460_000)
        raw_avail = props.at[item_id, "raw_available"] if item_id in props.index else pd.NA
        row.update(item_id=item_id, category_id=-1 if category_id is None else category_id, price=price,
                   price_is_real=is_real, available=bool(pd.isna(raw_avail) or raw_avail != "0"),
                   n_view=int(counts.at[item_id, "view"]) if item_id in counts.index else 0,
                   n_cart=int(counts.at[item_id, "addtocart"]) if item_id in counts.index else 0,
                   n_txn=int(counts.at[item_id, "transaction"]) if item_id in counts.index else 0)
        rows.append(row)
    return pd.DataFrame(rows)[COLUMNS]
