"""Item side information: categories, property values, and the category tree.

Three leakage rules govern this module:

* Rule 3 -- only records with ``timestamp <= T_train`` are admissible.
* Rule 4 -- an item may carry many records of the same attribute at different
  timestamps. The value used is the **latest record at or before T_train**, never
  the last line of the file. This is the easiest rule to get wrong.
* Rule 5 -- side information is attached only to items that exist in the train
  item mapping.

``item_properties`` holds 20.3M rows, so it is streamed twice instead of being
loaded. Pass 1 keeps a compact numeric view (``itemid``, property code, 64-bit
value hash, timestamp) to decide which PropertyValue nodes survive filtering;
pass 2 re-reads the file to materialise the string values of the survivors only.
Peak memory stays a few hundred MB (docs/DECISIONS.md muc D9).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.loader import iter_item_properties
from src.data.mapping import IdMapping
from src.data.splitter import TemporalSplit
from src.guards.leakage import assert_latest_record_selected
from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)

CATEGORY_PROPERTY = "categoryid"


@dataclass(frozen=True)
class SideInfo:
    """Side information tables, already restricted to train items and T_train.

    Attributes:
        item_category: ``[item_idx, category_id, category_idx, valid_from]`` --
            at most one row per item.
        item_property: ``[item_idx, pv_idx, valid_from]`` -- HAS_PROPERTY edges.
        property_values: ``[pv_idx, prop_key, prop_value, pv_id, freq]``.
        categories: ``[category_id, idx, depth, is_root]``.
        category_parent: ``[category_idx, parent_idx]`` -- PARENT_CATEGORY edges.
        stats: Counters reported in the audit table.
    """

    item_category: pd.DataFrame
    item_property: pd.DataFrame
    property_values: pd.DataFrame
    categories: pd.DataFrame
    category_parent: pd.DataFrame
    stats: dict[str, int]


def _hash_pv(prop: pd.Series, value: pd.Series) -> np.ndarray:
    """Hash a ``(property, value)`` pair into one uint64 key.

    The NUL separator keeps ``("a", "b::c")`` distinct from ``("a::b", "c")``.
    Collisions are checked for explicitly in :func:`extract_side_info`.
    """
    combined = prop.astype("str") + "\x00" + value.astype("str")
    return pd.util.hash_pandas_object(combined, index=False).to_numpy()


def _latest_per_group(keys: list[np.ndarray], timestamps: np.ndarray) -> np.ndarray:
    """Return positions of the newest row of each group.

    Sorting is by ``(*keys, timestamp)`` so the last row of each run is the most
    recent one. ``np.lexsort`` is stable, so equal timestamps resolve to the last
    occurrence in file order -- deterministic across runs.
    """
    order = np.lexsort((timestamps, *reversed(keys)))
    sorted_keys = [k[order] for k in keys]
    is_last = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        same_group = np.ones(len(order) - 1, dtype=bool)
        for k in sorted_keys:
            same_group &= k[:-1] == k[1:]
        is_last[:-1] = ~same_group
    return order[is_last]


def _category_depths(tree: pd.DataFrame) -> pd.DataFrame:
    """Compute depth of every category (root = 0) by walking parents."""
    parent = dict(
        zip(
            tree["categoryid"].to_numpy(),
            tree["parentid"].astype("float").to_numpy(),
            strict=True,
        )
    )
    depths: dict[int, int] = {}

    def depth_of(node: int) -> int:
        chain: list[int] = []
        current: int | None = node
        while current is not None and current not in depths:
            if current in chain:  # defensive: a cycle would loop forever
                raise ValueError(f"cycle in category tree at {current}")
            chain.append(current)
            parent_id = parent.get(current)
            current = None if parent_id is None or np.isnan(parent_id) else int(parent_id)
        base = 0 if current is None else depths[current] + 1
        for offset, node_id in enumerate(reversed(chain)):
            depths[node_id] = base + offset
        return depths[node]

    for node in tree["categoryid"].to_numpy():
        depth_of(int(node))
    return pd.DataFrame(
        {
            "category_id": np.array(list(depths), dtype="int32"),
            "depth": np.array(list(depths.values()), dtype="int32"),
        }
    )


def extract_side_info(
    cfg: Config,
    boundaries: TemporalSplit,
    mapping: IdMapping,
    category_tree: pd.DataFrame,
) -> SideInfo:
    """Build every side-information table for one cohort.

    Args:
        cfg: Resolved configuration (supplies the PV filters and T_train rules).
        boundaries: Temporal split; only ``t_train`` is used as the cutoff.
        mapping: Train-only visitor/item mapping.
        category_tree: Raw category hierarchy.

    Returns:
        A populated :class:`SideInfo`.

    Raises:
        RuntimeError: If a 64-bit hash collision is detected among surviving
            PropertyValue keys.
    """
    train_items = mapping.item_ids
    drop = set(cfg.graph.drop_properties)
    t_train = boundaries.t_train

    prop_codes: dict[str, int] = {}
    cat_item: list[np.ndarray] = []
    cat_value: list[np.ndarray] = []
    cat_ts: list[np.ndarray] = []
    pv_item: list[np.ndarray] = []
    pv_prop: list[np.ndarray] = []
    pv_hash: list[np.ndarray] = []
    pv_ts: list[np.ndarray] = []
    rows_seen = rows_after_cutoff = 0

    # ── Pass 1: compact numeric view ──────────────────────────────────────
    for chunk in iter_item_properties(cfg):
        rows_seen += len(chunk)
        chunk = chunk[chunk["timestamp"] <= t_train]
        rows_after_cutoff += len(chunk)
        if chunk.empty:
            continue
        chunk = chunk[_in_sorted(chunk["itemid"].to_numpy(), train_items)]
        if chunk.empty:
            continue

        is_category = chunk["property"] == CATEGORY_PROPERTY
        categories = chunk[is_category]
        if not categories.empty:
            cat_item.append(categories["itemid"].to_numpy())
            cat_value.append(pd.to_numeric(categories["value"], errors="coerce").to_numpy())
            cat_ts.append(categories["timestamp"].to_numpy())

        rest = chunk[~is_category & ~chunk["property"].isin(drop)]
        if rest.empty:
            continue
        for name in rest["property"].unique():
            prop_codes.setdefault(name, len(prop_codes))
        pv_item.append(rest["itemid"].to_numpy())
        pv_prop.append(rest["property"].map(prop_codes).to_numpy().astype("int32"))
        pv_hash.append(_hash_pv(rest["property"], rest["value"]))
        pv_ts.append(rest["timestamp"].to_numpy())

    log.info(
        "pass 1: %s dong, %s dong <= T_train, %s dong PV cua item trong train",
        f"{rows_seen:,}", f"{rows_after_cutoff:,}", f"{sum(len(a) for a in pv_item):,}",
    )

    # ── Rule 4: newest admissible record per (item, property) ─────────────
    item_category = _resolve_categories(cat_item, cat_value, cat_ts, mapping)
    pv_table = _resolve_property_values(pv_item, pv_prop, pv_hash, pv_ts, mapping)

    # ── PV filtering: freq >= min_pv_freq, then top max_property_nodes ─────
    keys, freq = np.unique(pv_table["key"], return_counts=True)
    keep = freq >= cfg.graph.min_pv_freq
    keys, freq = keys[keep], freq[keep]
    n_after_freq = len(keys)
    if len(keys) > cfg.graph.max_property_nodes:
        top = np.lexsort((keys, -freq))[: cfg.graph.max_property_nodes]
        keys, freq = keys[top], freq[top]
    order = np.argsort(keys)
    keys, freq = keys[order], freq[order]
    log.info(
        "PV: %s phan biet -> %s co freq >= %d -> giu %s node",
        f"{len(np.unique(pv_table['key'])):,}", f"{n_after_freq:,}",
        cfg.graph.min_pv_freq, f"{len(keys):,}",
    )

    # ── Pass 2: materialise strings for surviving keys only ───────────────
    labels = _resolve_pv_labels(cfg, t_train, train_items, drop, keys)
    if len(labels) != len(keys):
        raise RuntimeError(
            f"64-bit hash collision: {len(keys)} keys resolved to {len(labels)} distinct pairs"
        )

    surviving = _in_sorted(pv_table["key"], keys)
    item_property = pd.DataFrame(
        {
            "item_idx": pv_table["item_idx"][surviving],
            "pv_idx": np.searchsorted(keys, pv_table["key"][surviving]).astype("int32"),
            "valid_from": pv_table["valid_from"][surviving],
        }
    )
    property_values = pd.DataFrame(
        {
            "pv_idx": np.arange(len(keys), dtype="int32"),
            "prop_key": [labels[k][0] for k in keys],
            "prop_value": [labels[k][1] for k in keys],
            "freq": freq.astype("int32"),
        }
    )
    property_values["pv_id"] = property_values["prop_key"] + "::" + property_values["prop_value"]

    categories, category_parent = _build_category_nodes(category_tree, item_category)
    item_category = item_category.merge(
        categories[["category_id", "idx"]], on="category_id", how="inner"
    )

    stats = {
        "property_rows_total": rows_seen,
        "property_rows_within_cutoff": rows_after_cutoff,
        "n_item_category_edges": len(item_category),
        "n_item_property_edges": len(item_property),
        "n_property_values": len(property_values),
        "n_categories": len(categories),
        "n_category_parent_edges": len(category_parent),
        "n_items_with_category": int(item_category["item_idx"].nunique()),
        "n_items_with_property": int(item_property["item_idx"].nunique()),
    }
    return SideInfo(
        item_category=item_category[["item_idx", "category_id", "idx", "valid_from"]].rename(
            columns={"idx": "category_idx"}
        ),
        item_property=item_property,
        property_values=property_values,
        categories=categories,
        category_parent=category_parent,
        stats=stats,
    )


def _in_sorted(values: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """Boolean mask of ``values`` present in the sorted ``universe``."""
    pos = np.searchsorted(universe, values)
    pos = np.clip(pos, 0, len(universe) - 1)
    return universe[pos] == values


def _resolve_categories(
    items: list[np.ndarray],
    values: list[np.ndarray],
    stamps: list[np.ndarray],
    mapping: IdMapping,
) -> pd.DataFrame:
    """Keep the newest ``categoryid`` record per item (leakage rule 4)."""
    if not items:
        return pd.DataFrame({"item_idx": [], "category_id": []}).astype("int32")
    item_ids = np.concatenate(items)
    category_ids = np.concatenate(values)
    timestamps = np.concatenate(stamps)
    valid = ~np.isnan(category_ids)
    item_ids, category_ids, timestamps = item_ids[valid], category_ids[valid], timestamps[valid]

    last = _latest_per_group([item_ids], timestamps)
    assert_latest_record_selected([item_ids], timestamps, last, rule="4 (item-category)")
    return pd.DataFrame(
        {
            "item_idx": np.searchsorted(mapping.item_ids, item_ids[last]).astype("int32"),
            "category_id": category_ids[last].astype("int32"),
            "valid_from": timestamps[last].astype("int64"),
        }
    )


def _resolve_property_values(
    items: list[np.ndarray],
    props: list[np.ndarray],
    hashes: list[np.ndarray],
    stamps: list[np.ndarray],
    mapping: IdMapping,
) -> dict[str, np.ndarray]:
    """Keep the newest value per ``(item, property)`` (leakage rule 4).

    ``item_idx`` is the *matrix index* from the train mapping, never the raw
    ``itemid``: the graph layer indexes rows by position.
    """
    item_ids = np.concatenate(items)
    prop_codes = np.concatenate(props)
    keys = np.concatenate(hashes)
    timestamps = np.concatenate(stamps)
    last = _latest_per_group([item_ids, prop_codes], timestamps)
    assert_latest_record_selected([item_ids, prop_codes], timestamps, last, rule="4 (item-property)")
    log.info(
        "rule 4: %s dong PV -> %s cap (item, property) duy nhat",
        f"{len(item_ids):,}", f"{len(last):,}",
    )
    return {
        "item_idx": np.searchsorted(mapping.item_ids, item_ids[last]).astype("int32"),
        "key": keys[last],
        "valid_from": timestamps[last].astype("int64"),
    }


def _resolve_pv_labels(
    cfg: Config,
    t_train: int,
    train_items: np.ndarray,
    drop: set[str],
    keys: np.ndarray,
) -> dict[int, tuple[str, str]]:
    """Second pass: recover ``(property, value)`` strings for surviving keys."""
    labels: dict[int, tuple[str, str]] = {}
    for chunk in iter_item_properties(cfg):
        chunk = chunk[chunk["timestamp"] <= t_train]
        if chunk.empty:
            continue
        chunk = chunk[_in_sorted(chunk["itemid"].to_numpy(), train_items)]
        chunk = chunk[(chunk["property"] != CATEGORY_PROPERTY) & ~chunk["property"].isin(drop)]
        if chunk.empty:
            continue
        chunk_keys = _hash_pv(chunk["property"], chunk["value"])
        wanted = _in_sorted(chunk_keys, keys)
        if not wanted.any():
            continue
        subset = chunk[wanted]
        for key, prop, value in zip(
            chunk_keys[wanted], subset["property"], subset["value"], strict=True
        ):
            labels.setdefault(int(key), (str(prop), str(value)))
        if len(labels) == len(keys):
            break
    return labels


def _build_category_nodes(
    tree: pd.DataFrame, item_category: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Category nodes (used categories plus ancestors) and PARENT edges."""
    parent = dict(
        zip(tree["categoryid"].to_numpy(), tree["parentid"].astype("float").to_numpy(), strict=True)
    )
    used = set(int(c) for c in item_category["category_id"].unique())
    in_tree = {c for c in used if c in parent}

    keep: set[int] = set()
    for node in in_tree:
        current: int | None = node
        while current is not None and current not in keep:
            keep.add(current)
            parent_id = parent.get(current)
            current = None if parent_id is None or np.isnan(parent_id) else int(parent_id)

    log.info(
        "category: %s gia tri tren item, %s noi duoc vao tree (%.2f%%), %s node ke ca to tien",
        f"{len(used):,}", f"{len(in_tree):,}",
        100 * len(in_tree) / max(len(used), 1), f"{len(keep):,}",
    )

    kept = np.sort(np.array(sorted(keep), dtype="int32"))
    depths = _category_depths(tree[tree["categoryid"].isin(kept)])
    categories = pd.DataFrame({"category_id": kept, "idx": np.arange(len(kept), dtype="int32")})
    categories = categories.merge(depths, on="category_id", how="left")
    categories["depth"] = categories["depth"].fillna(0).astype("int32")
    categories["is_root"] = [
        parent.get(int(c)) is None or np.isnan(parent.get(int(c), np.nan)) for c in kept
    ]

    child_idx: list[int] = []
    parent_idx: list[int] = []
    index_of = {int(c): i for i, c in enumerate(kept)}
    for category_id, position in index_of.items():
        parent_id = parent.get(category_id)
        if parent_id is not None and not np.isnan(parent_id) and int(parent_id) in index_of:
            child_idx.append(position)
            parent_idx.append(index_of[int(parent_id)])
    category_parent = pd.DataFrame(
        {
            "category_idx": np.array(child_idx, dtype="int32"),
            "parent_idx": np.array(parent_idx, dtype="int32"),
        }
    )
    return categories, category_parent
