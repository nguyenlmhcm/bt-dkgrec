"""Buoc 10 — xuat lop truy vet sang Neo4j (docs/KG_DESIGN.md muc 1-3, 7).

Kien truc hai lop
-----------------
Lop CHIEU (`scipy.sparse` -> torch) la thu mo hinh hoc tren: moi cap
``(visitor, item)`` la MOT canh mang trong so tong hop ``W(u,i)``.
Lop TRUY VET (Neo4j) la thu con nguoi doc: moi su kien la mot node ``:Event``
rieng, nen demo tra loi duoc "vi sao item nay duoc goi y cho khach nay".

Hai lop phai sinh tu CUNG tap train va CUNG ``T_train``. Neu lech, demo se giai
thich mot mo hinh khac voi mo hinh da bao cao ket qua -- va khong ai phat hien
ra, vi ca hai deu chay tot. :func:`assert_layers_consistent` chan dung dieu do:
tong trong so su kien cua moi cap phai bang trong so canh da chieu.

Cong thuc trong so KHONG duoc cai lai o day. Module nay goi dung
``weighting_for_model()`` ma Buoc 4 dung -- mot cai dat, mot su that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.graph.weighting import event_age_days, weighting_for_model
from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)

#: Sai so tuong doi cho phep giua hai lop. Cong don float64 tren hang trieu canh
#: khong bao gio khop tuyet doi; 1e-9 van chat hon moi sai lech co y nghia.
LAYER_TOLERANCE = 1e-9

#: Ten file CSV. Giu on dinh vi `import.cypher` tham chieu truc tiep.
NODE_FILES = ("visitors", "items", "categories", "property_values", "events")
EDGE_FILES = ("interacted_with", "performed_targets", "has_category",
              "has_property", "parent_category")


class LayerMismatchError(RuntimeError):
    """Lop truy vet va lop chieu khong sinh ra cung mot do thi."""


def build_event_layer(
    train_events: pd.DataFrame, cfg: Config, t_train: int
) -> pd.DataFrame:
    """Tinh trong so cho TUNG su kien -- day la thu lop chieu da gop mat.

    Args:
        train_events: Su kien train, da co ``visitor_idx``/``item_idx``.
        cfg: Cau hinh -- quyet dinh chien luoc trong so nao duoc dung.
        t_train: Moc quan sat tau, tinh bang ms.

    Returns:
        Ban sao cua ``train_events`` them cot ``w_event`` va ``event_id``.
    """
    weighting = weighting_for_model(cfg)
    behaviors = tuple(cfg.data.history_behaviors)
    codes = pd.Categorical(train_events["behavior"], categories=behaviors).codes
    if (codes < 0).any():
        unknown = set(train_events["behavior"]) - set(behaviors)
        raise ValueError(f"hanh vi la: {sorted(unknown)}")

    ages = event_age_days(
        train_events["timestamp"].to_numpy(), t_train, cfg.weighting.d_day
    )
    out = train_events.copy()
    out["w_event"] = weighting.edge_weight(codes.astype("int64"), ages)
    out["event_id"] = (
        out["visitorid"].astype(str) + "_"
        + out["itemid"].astype(str) + "_"
        + out["timestamp"].astype(str)
    )
    return out


def assert_layers_consistent(
    event_layer: pd.DataFrame, projected: pd.DataFrame, tolerance: float = LAYER_TOLERANCE
) -> dict[str, object]:
    """Tong trong so su kien moi cap phai bang trong so canh da chieu.

    Guard nay duoc dat ten trong docs/KG_DESIGN.md muc 1 va la ly do ca hai lop
    dung duoc canh nhau. No bat ba loai lech cung luc: khac tap train, khac
    ``T_train``, va khac chien luoc trong so.

    Raises:
        LayerMismatchError: Khi so cap lech nhau hoac trong so lech qua nguong.
    """
    summed = (
        event_layer.groupby(["visitor_idx", "item_idx"], sort=True)["w_event"]
        .sum()
        .rename("w_trace")
    )
    merged = projected.set_index(["visitor_idx", "item_idx"]).join(summed, how="outer")

    missing_trace = int(merged["w_trace"].isna().sum())
    missing_projected = int(merged["weight"].isna().sum())
    if missing_trace or missing_projected:
        raise LayerMismatchError(
            f"hai lop khong cung tap canh: {missing_trace:,} cap chi co o lop chieu, "
            f"{missing_projected:,} cap chi co o lop truy vet"
        )

    denominator = merged["weight"].abs().clip(lower=1e-12)
    relative = ((merged["w_trace"] - merged["weight"]).abs() / denominator)
    worst = float(relative.max())
    if worst > tolerance:
        row = merged.loc[relative.idxmax()]
        raise LayerMismatchError(
            f"trong so hai lop lech toi da {worst:.3e} (nguong {tolerance:.0e}); "
            f"cap te nhat: chieu={row['weight']:.9f} truy_vet={row['w_trace']:.9f}"
        )

    record = {
        "n_pairs": int(len(merged)),
        "n_events": int(len(event_layer)),
        "worst_relative_difference": worst,
        "tolerance": tolerance,
    }
    log.info(
        "hai lop khop: %s cap, %s su kien, lech toi da %.3e",
        f"{record['n_pairs']:,}", f"{record['n_events']:,}", worst,
    )
    return record


def select_visitors(
    events: pd.DataFrame,
    scope: str,
    target_behaviors: tuple[str, ...],
    split: str = "test",
) -> np.ndarray | None:
    """Tap visitor duoc xuat. ``None`` nghia la xuat het.

    ``"evaluated"`` phai trung DUNG quan the ma evaluator bao cao: nguoi dung co
    it nhat mot HANH VI MUC TIEU trong split danh gia. Loc theo moi su kien se
    ra 275.826 visitor thay vi 593 -- demo se minh hoa mot tap khac voi bang ket
    qua, va nguoi xem khong the doi chieu hai thu voi nhau.

    Chi visitor warm song sot sau buoc nay, vi lop truy vet dung tren su kien
    TRAIN: nguoi khong co lich su train thi khong co canh nao de truy vet.

    Args:
        events: Toan bo su kien, con cot ``split``.
        scope: ``"evaluated"`` hoac ``"all"``.
        target_behaviors: ``cfg.data.target_behaviors`` -- dung mot nguon voi evaluator.
        split: Split danh gia, mac dinh ``"test"``.
    """
    if scope == "all":
        return None
    if scope != "evaluated":
        raise ValueError(f"scope la: {scope!r}; chi 'all' hoac 'evaluated'")
    chosen = events.loc[
        (events["split"] == split) & events["behavior"].isin(target_behaviors),
        "visitorid",
    ].unique()
    return np.sort(chosen)


def _write(frame: pd.DataFrame, path: Path) -> int:
    frame.to_csv(path, index=False)
    log.info("  %-24s %s dong", path.name, f"{len(frame):,}")
    return len(frame)


def export_csv(
    cfg: Config,
    interim_dir: Path,
    graph_dir: Path,
    out_dir: Path,
    scope: str = "evaluated",
) -> dict[str, object]:
    """Xuat ca hai lop ra CSV, kem `import.cypher`.

    Thu tu bat buoc: dung lop truy vet -> DOI CHIEU voi lop chieu -> moi ghi.
    Ghi truoc roi kiem sau thi khi guard no da co mot thu muc CSV sai nam tren
    dia cho ai do nhap vao Neo4j.

    Args:
        cfg: Cau hinh cua cohort/model dang xuat.
        interim_dir: ``data/interim/<cohort>/``.
        graph_dir: ``data/processed/<cohort>/<model>/``.
        out_dir: Thu muc dich cho CSV.
        scope: ``"evaluated"`` (chi visitor co trong tap danh gia) hoac ``"all"``.

    Returns:
        Thong ke ghi ra, kem ban ghi cua guard hai lop.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(interim_dir / "events.parquet")
    visitors = pd.read_parquet(interim_dir / "visitors.parquet")
    items = pd.read_parquet(interim_dir / "items.parquet")
    t_train = int(json.loads((interim_dir / "split.json").read_text())["t_train"])

    train = events[events["split"] == "train"]
    train = train.merge(visitors, left_on="visitorid", right_on="visitor_id")
    train = train.rename(columns={"idx": "visitor_idx"})
    train = train.merge(items, left_on="itemid", right_on="item_id")
    train = train.rename(columns={"idx": "item_idx"})

    layer = build_event_layer(train, cfg, t_train)
    projected = pd.read_parquet(graph_dir / "edges_interacted.parquet")

    # Doi chieu tren TOAN BO do thi, khong phai tren phan sap xuat. Mot tap con
    # khop nhau khong chung minh duoc gi ve phan con lai.
    consistency = assert_layers_consistent(layer, projected)

    keep = select_visitors(events, scope, tuple(cfg.data.target_behaviors))
    if keep is not None:
        layer = layer[layer["visitorid"].isin(keep)]
        kept_items = layer["item_idx"].unique()
        projected = projected[
            projected["visitor_idx"].isin(layer["visitor_idx"].unique())
            & projected["item_idx"].isin(kept_items)
        ]
    else:
        kept_items = projected["item_idx"].unique()

    item_id_of = items.set_index("idx")["item_id"]
    visitor_id_of = visitors.set_index("idx")["visitor_id"]

    counts: dict[str, int] = {}

    # ── Node ────────────────────────────────────────────────────────────
    stats = events.groupby("visitorid")["timestamp"].agg(["min", "max", "count"])
    warm = set(visitors["visitor_id"])
    node_visitors = pd.DataFrame({
        "visitor_id": layer["visitorid"].unique() if keep is not None else visitors["visitor_id"],
    })
    node_visitors = node_visitors.join(stats, on="visitor_id")
    node_visitors["idx"] = node_visitors["visitor_id"].map(
        visitors.set_index("visitor_id")["idx"]).fillna(-1).astype("int64")
    node_visitors["segment"] = np.where(
        node_visitors["visitor_id"].isin(warm), "warm", "cold")
    node_visitors = node_visitors.rename(
        columns={"min": "first_seen", "max": "last_seen", "count": "n_events"})
    counts["visitors"] = _write(node_visitors, out_dir / "visitors.csv")

    item_stats = events.groupby("itemid")["timestamp"].agg(["min", "max", "count"])
    node_items = pd.DataFrame({"idx": np.sort(kept_items)})
    node_items["item_id"] = node_items["idx"].map(item_id_of)
    node_items = node_items.join(item_stats, on="item_id")
    node_items = node_items.rename(
        columns={"min": "first_seen", "max": "last_seen", "count": "n_interactions"})
    counts["items"] = _write(node_items, out_dir / "items.csv")

    categories = pd.read_parquet(interim_dir / "side_categories.parquet")
    counts["categories"] = _write(categories, out_dir / "categories.csv")

    property_values = pd.read_parquet(interim_dir / "side_property_values.parquet")
    counts["property_values"] = _write(property_values, out_dir / "property_values.csv")

    node_events = layer[["event_id", "visitorid", "itemid", "behavior",
                         "timestamp", "w_event"]]
    counts["events"] = _write(node_events, out_dir / "events.csv")

    # ── Canh ────────────────────────────────────────────────────────────
    edges = projected.copy()
    edges["visitor_id"] = edges["visitor_idx"].map(visitor_id_of)
    edges["item_id"] = edges["item_idx"].map(item_id_of)
    counts["interacted_with"] = _write(
        edges[["visitor_id", "item_id", "weight", "n_view", "n_cart", "n_txn", "last_ts"]],
        out_dir / "interacted_with.csv")

    has_category = pd.read_parquet(interim_dir / "side_item_category.parquet")
    has_category = has_category[has_category["item_idx"].isin(kept_items)].copy()
    has_category["item_id"] = has_category["item_idx"].map(item_id_of)
    has_category["rel_weight"] = 1.0
    counts["has_category"] = _write(
        has_category[["item_id", "category_id", "valid_from", "rel_weight"]],
        out_dir / "has_category.csv")

    has_property = pd.read_parquet(interim_dir / "side_item_property.parquet")
    has_property = has_property[has_property["item_idx"].isin(kept_items)].copy()
    has_property["item_id"] = has_property["item_idx"].map(item_id_of)
    has_property["pv_id"] = has_property["pv_idx"].map(
        property_values.set_index("pv_idx")["pv_id"])
    has_property["rel_weight"] = 1.0
    counts["has_property"] = _write(
        has_property[["item_id", "pv_id", "valid_from", "rel_weight"]],
        out_dir / "has_property.csv")

    parents = pd.read_parquet(interim_dir / "side_category_parent.parquet").copy()
    id_of_category = categories.set_index("idx")["category_id"]
    parents["category_id"] = parents["category_idx"].map(id_of_category)
    parents["parent_id"] = parents["parent_idx"].map(id_of_category)
    parents["rel_weight"] = 1.0
    counts["parent_category"] = _write(
        parents[["category_id", "parent_id", "rel_weight"]],
        out_dir / "parent_category.csv")

    (out_dir / "import.cypher").write_text(IMPORT_CYPHER, encoding="utf-8")

    manifest = {
        "cohort": cfg.cohort.name,
        "model": cfg.model.name,
        "scope": scope,
        "t_train": t_train,
        "weighting": weighting_for_model(cfg).describe(),
        "layer_consistency": consistency,
        "counts": counts,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


#: DDL + LOAD CSV. Rang buoc UNIQUE theo docs/KG_DESIGN.md muc 7 -- chung vua
#: chan trung lap vua tao index cho MATCH trong buoc nap canh, nen phai chay
#: TRUOC moi LOAD CSV chu khong phai sau.
IMPORT_CYPHER = """// BT-DKGRec-GCN — nap lop truy vet vao Neo4j
// Dat toan bo file CSV vao thu muc import/ cua Neo4j roi chay file nay.

CREATE CONSTRAINT visitor_id IF NOT EXISTS
  FOR (v:Visitor) REQUIRE v.visitor_id IS UNIQUE;
CREATE CONSTRAINT item_id IF NOT EXISTS
  FOR (i:Item) REQUIRE i.item_id IS UNIQUE;
CREATE CONSTRAINT category_id IF NOT EXISTS
  FOR (c:Category) REQUIRE c.category_id IS UNIQUE;
CREATE CONSTRAINT pv_id IF NOT EXISTS
  FOR (p:PropertyValue) REQUIRE p.pv_id IS UNIQUE;
CREATE CONSTRAINT event_id IF NOT EXISTS
  FOR (e:Event) REQUIRE e.event_id IS UNIQUE;
CREATE INDEX visitor_segment IF NOT EXISTS FOR (v:Visitor) ON (v.segment);
CREATE INDEX item_idx IF NOT EXISTS FOR (i:Item) ON (i.idx);

LOAD CSV WITH HEADERS FROM 'file:///visitors.csv' AS row
CALL { WITH row
  CREATE (:Visitor {visitor_id: toInteger(row.visitor_id), idx: toInteger(row.idx),
    first_seen: toInteger(row.first_seen), last_seen: toInteger(row.last_seen),
    n_events: toInteger(row.n_events), segment: row.segment})
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///items.csv' AS row
CALL { WITH row
  CREATE (:Item {item_id: toInteger(row.item_id), idx: toInteger(row.idx),
    first_seen: toInteger(row.first_seen), last_seen: toInteger(row.last_seen),
    n_interactions: toInteger(row.n_interactions)})
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///categories.csv' AS row
CALL { WITH row
  CREATE (:Category {category_id: toInteger(row.category_id), idx: toInteger(row.idx),
    depth: toInteger(row.depth), is_root: toBoolean(row.is_root)})
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///property_values.csv' AS row
CALL { WITH row
  CREATE (:PropertyValue {pv_id: row.pv_id, prop_key: row.prop_key,
    prop_value: row.prop_value, idx: toInteger(row.pv_idx)})
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///events.csv' AS row
CALL { WITH row
  MATCH (v:Visitor {visitor_id: toInteger(row.visitorid)})
  MATCH (i:Item {item_id: toInteger(row.itemid)})
  CREATE (e:Event {event_id: row.event_id, behavior: row.behavior,
    timestamp: toInteger(row.timestamp), w_event: toFloat(row.w_event)})
  CREATE (v)-[:PERFORMED]->(e)
  CREATE (e)-[:TARGETS]->(i)
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///interacted_with.csv' AS row
CALL { WITH row
  MATCH (v:Visitor {visitor_id: toInteger(row.visitor_id)})
  MATCH (i:Item {item_id: toInteger(row.item_id)})
  CREATE (v)-[:INTERACTED_WITH {weight: toFloat(row.weight),
    n_view: toInteger(row.n_view), n_cart: toInteger(row.n_cart),
    n_txn: toInteger(row.n_txn), last_ts: toInteger(row.last_ts)}]->(i)
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///has_category.csv' AS row
CALL { WITH row
  MATCH (i:Item {item_id: toInteger(row.item_id)})
  MATCH (c:Category {category_id: toInteger(row.category_id)})
  CREATE (i)-[:HAS_CATEGORY {valid_from: toInteger(row.valid_from),
    rel_weight: toFloat(row.rel_weight)}]->(c)
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///has_property.csv' AS row
CALL { WITH row
  MATCH (i:Item {item_id: toInteger(row.item_id)})
  MATCH (p:PropertyValue {pv_id: row.pv_id})
  CREATE (i)-[:HAS_PROPERTY {valid_from: toInteger(row.valid_from),
    rel_weight: toFloat(row.rel_weight)}]->(p)
} IN TRANSACTIONS OF 10000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///parent_category.csv' AS row
CALL { WITH row
  MATCH (c:Category {category_id: toInteger(row.category_id)})
  MATCH (p:Category {category_id: toInteger(row.parent_id)})
  CREATE (c)-[:PARENT_CATEGORY {rel_weight: toFloat(row.rel_weight)}]->(p)
} IN TRANSACTIONS OF 10000 ROWS;
"""
