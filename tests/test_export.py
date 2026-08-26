"""Tests cho Buoc 10 — xuat lop truy vet sang Neo4j.

Dieu duoc bao ve o day khong phai "CSV co ghi ra khong" ma la: **demo giai thich
dung mo hinh da bao cao ket qua**. Neu lop truy vet lech khoi lop chieu, demo se
noi mot dang va bang ket qua noi mot dang khac, ma ca hai deu chay tot.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.export.neo4j_export import (
    IMPORT_CYPHER,
    LayerMismatchError,
    assert_layers_consistent,
    build_event_layer,
    select_visitors,
)
from src.utils.config import load_config

DAY = 86_400_000
T_TRAIN = 100 * DAY


def _events() -> pd.DataFrame:
    """Hai visitor; visitor 1 co ba hanh vi tren cung mot item."""
    rows = [
        (1, 10, "view", 90 * DAY, "train", 0, 0),
        (1, 10, "addtocart", 95 * DAY, "train", 0, 0),
        (1, 11, "transaction", 100 * DAY, "train", 0, 1),
        (2, 10, "view", 50 * DAY, "train", 1, 0),
    ]
    frame = pd.DataFrame(rows, columns=["visitorid", "itemid", "behavior",
                                        "timestamp", "split", "visitor_idx", "item_idx"])
    return frame


def _projected(layer: pd.DataFrame) -> pd.DataFrame:
    return (layer.groupby(["visitor_idx", "item_idx"], as_index=False)["w_event"]
            .sum().rename(columns={"w_event": "weight"}))


# ── Trong so tung su kien ────────────────────────────────────────────────


def test_event_weight_follows_formula_3_17() -> None:
    """w = alpha_b * exp(-lambda * dt), khong duoc cai lai o module export."""
    cfg = load_config(model="bt_dkgrec", cohort="original")
    layer = build_event_layer(_events(), cfg, T_TRAIN)

    alpha = cfg.weighting.alpha
    lam = cfg.weighting.lambda_decay
    row = layer.iloc[0]                      # view, 10 ngay tuoi
    assert row["w_event"] == pytest.approx(alpha["view"] * math.exp(-lam * 10))

    txn = layer.iloc[2]                      # transaction, 0 ngay tuoi
    assert txn["w_event"] == pytest.approx(alpha["transaction"])


def test_static_kg_gcn_gives_every_event_weight_one() -> None:
    """Ablation: bo ca tin hieu hanh vi lan suy giam thoi gian."""
    cfg = load_config(model="static_kg_gcn", cohort="original")
    layer = build_event_layer(_events(), cfg, T_TRAIN)

    assert (layer["w_event"] == 1.0).all()


def test_event_id_is_unique_per_event() -> None:
    cfg = load_config(model="bt_dkgrec", cohort="original")
    layer = build_event_layer(_events(), cfg, T_TRAIN)

    assert layer["event_id"].is_unique


def test_an_unknown_behavior_is_refused_rather_than_silently_coded() -> None:
    cfg = load_config(model="bt_dkgrec", cohort="original")
    events = _events()
    events.loc[0, "behavior"] = "wishlist"

    with pytest.raises(ValueError, match="wishlist"):
        build_event_layer(events, cfg, T_TRAIN)


# ── Guard hai lop ────────────────────────────────────────────────────────


def test_matching_layers_pass_the_guard() -> None:
    cfg = load_config(model="bt_dkgrec", cohort="original")
    layer = build_event_layer(_events(), cfg, T_TRAIN)

    record = assert_layers_consistent(layer, _projected(layer))

    assert record["n_pairs"] == 3           # (1,10), (1,11), (2,10)
    assert record["n_events"] == 4
    assert record["worst_relative_difference"] == pytest.approx(0.0)


def test_a_drifted_weight_is_caught() -> None:
    """★ Neu lop chieu dung T_train khac, trong so se lech — demo se noi sai."""
    cfg = load_config(model="bt_dkgrec", cohort="original")
    layer = build_event_layer(_events(), cfg, T_TRAIN)
    projected = _projected(layer)
    projected.loc[0, "weight"] *= 1.01      # lech 1%

    with pytest.raises(LayerMismatchError, match="lech toi da"):
        assert_layers_consistent(layer, projected)


def test_a_pair_present_in_only_one_layer_is_caught() -> None:
    cfg = load_config(model="bt_dkgrec", cohort="original")
    layer = build_event_layer(_events(), cfg, T_TRAIN)
    projected = _projected(layer).iloc[:-1]     # bo mot cap

    with pytest.raises(LayerMismatchError, match="khong cung tap canh"):
        assert_layers_consistent(layer, projected)


# ── Chon visitor cho demo ────────────────────────────────────────────────


def _split_events() -> pd.DataFrame:
    return pd.DataFrame([
        (1, 10, "view", "test"),          # chi xem -> KHONG phai nguoi duoc danh gia
        (2, 10, "addtocart", "test"),     # hanh vi muc tieu -> duoc danh gia
        (3, 11, "transaction", "test"),
        (4, 12, "addtocart", "train"),    # sai split
    ], columns=["visitorid", "itemid", "behavior", "split"])


def test_evaluated_scope_matches_the_evaluator_population() -> None:
    """★ Loc theo MOI su kien se ra 275.826 visitor thay vi 593.

    Demo phai minh hoa dung quan the ma bang ket qua noi ve, neu khong nguoi xem
    khong doi chieu duoc hai thu voi nhau.
    """
    chosen = select_visitors(_split_events(), "evaluated",
                             target_behaviors=("addtocart", "transaction"))

    assert chosen.tolist() == [2, 3]


def test_all_scope_exports_everything() -> None:
    assert select_visitors(_split_events(), "all", ("addtocart",)) is None


def test_an_unknown_scope_is_refused() -> None:
    with pytest.raises(ValueError, match="scope"):
        select_visitors(_split_events(), "mot-nua", ("addtocart",))


# ── Kich ban nap Neo4j ───────────────────────────────────────────────────


def test_constraints_are_created_before_any_load() -> None:
    """Rang buoc UNIQUE cung tao index; tao sau LOAD CSV thi MATCH quet toan bang."""
    first_load = IMPORT_CYPHER.index("LOAD CSV")
    for constraint in ("visitor_id", "item_id", "category_id", "pv_id", "event_id"):
        position = IMPORT_CYPHER.index(f"CREATE CONSTRAINT {constraint}")
        assert position < first_load, f"rang buoc {constraint} tao sau LOAD CSV"


def test_every_node_and_edge_type_of_the_design_is_loaded() -> None:
    """docs/KG_DESIGN.md muc 2-3 liet ke 5 node va 6 quan he."""
    for label in ("Visitor", "Item", "Category", "PropertyValue", "Event"):
        assert f":{label} " in IMPORT_CYPHER or f":{label} {{" in IMPORT_CYPHER
    for relation in ("INTERACTED_WITH", "PERFORMED", "TARGETS",
                     "HAS_CATEGORY", "HAS_PROPERTY", "PARENT_CATEGORY"):
        assert relation in IMPORT_CYPHER
