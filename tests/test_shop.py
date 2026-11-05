"""Cua hang demo (D39): ten minh hoa on dinh, giu cau truc that, giai thich dung do thi."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.shop.catalog import SHOWCASE, describe_item, persona_name, round_price, stable_hash
from src.shop.names import COLOURS, DEPARTMENTS, NO_COLOUR

ROOT = Path(__file__).resolve().parents[1]


def test_stable_hash_does_not_depend_on_the_process() -> None:
    assert stable_hash("type", 320) == stable_hash("type", 320)
    assert stable_hash("a", 1) != stable_hash("a", 2)


def test_items_of_one_category_share_a_product_type() -> None:
    """★ Giu cau truc that: cung danh muc goc thi cung loai san pham, chi khac bien the."""
    types = {describe_item(item, 1, 320)["product_type"] for item in range(1000, 1200)}
    assert len(types) == 1


def test_a_product_title_never_names_two_colours() -> None:
    for dept_idx, (department, *_rest) in enumerate(DEPARTMENTS):
        for item in range(300):
            row = describe_item(item, dept_idx, item % 40)
            assert row["name"].count("màu ") <= 1
            if "màu" in row["product_type"]:
                continue
            if department in NO_COLOUR:
                assert "màu " not in row["name"]
            else:
                assert f"màu {COLOURS[row['tone']]}" in row["name"]


def test_every_department_has_enough_plain_variants() -> None:
    for _department, _icon, types, variants in DEPARTMENTS:
        assert types and len([v for v in variants if not v.startswith("màu")]) >= 4


def test_uncategorised_items_keep_their_id_visible() -> None:
    row = describe_item(123, None, None)
    assert "123" in row["name"] and row["dept_idx"] == -1


@pytest.mark.parametrize("value,expected", [(257160.0, 257000), (5400.0, 15000), (5e8, 9000000)])
def test_price_is_read_as_vnd_within_shop_bounds(value: float, expected: int) -> None:
    assert round_price(value) == expected


def test_showcase_names_do_not_repeat() -> None:
    assert len(set(SHOWCASE.values())) == len(SHOWCASE)
    assert all(persona_name(v) == name for v, name in SHOWCASE.items())


def _catalog() -> dict[int, dict]:
    base = {"name": "", "department": "D", "product_type": "T"}
    return {1: {**base, "item_id": 1, "category_id": 10, "dept_idx": 0}, 2: {**base, "item_id": 2, "category_id": 10, "dept_idx": 0},
            3: {**base, "item_id": 3, "category_id": 11, "dept_idx": 0}, 4: {**base, "item_id": 4, "category_id": 12, "dept_idx": 5},
            5: {**base, "item_id": 5, "category_id": -1, "dept_idx": -1}}


def _history(*rows: tuple[int, str]) -> pd.DataFrame:
    return pd.DataFrame({"item_id": [r[0] for r in rows], "behavior": [r[1] for r in rows],
                         "timestamp": list(range(len(rows)))})


def test_explanation_prefers_the_same_category() -> None:
    from src.shop.engine import explain

    catalog = _catalog()
    why = explain(_history((3, "view"), (2, "transaction")), catalog[1], catalog)
    assert why["kind"] == "category" and why["seen"]["item_id"] == 2


def test_explanation_falls_back_to_department_then_collaborative() -> None:
    from src.shop.engine import explain

    catalog = _catalog()
    assert explain(_history((3, "view")), catalog[1], catalog)["kind"] == "department"
    assert explain(_history((4, "view")), catalog[1], catalog)["kind"] == "collaborative"


def test_explanation_reports_a_direct_touch() -> None:
    from src.shop.engine import explain

    catalog = _catalog()
    assert explain(_history((1, "addtocart")), catalog[1], catalog)["kind"] == "direct"


def test_uncategorised_items_are_never_explained_by_category() -> None:
    from src.shop.engine import explain

    catalog = _catalog()
    assert explain(_history((5, "view")), {**catalog[5], "item_id": 99}, catalog)["kind"] == "collaborative"


def test_cart_records_survive_a_clear_only_when_asked(tmp_path: Path) -> None:
    from src.shop.db import ShopDB

    db = ShopDB(tmp_path / "s.sqlite")
    order = db.run("INSERT INTO orders(visitor, total, created) VALUES (?,?,?)", 1, 100, 0.0)
    assert order == 1 and db.one("SELECT total FROM orders")["total"] == 100
    db.clear()
    assert db.rows("SELECT * FROM orders") == []
    assert db.run("INSERT INTO orders(visitor, total, created) VALUES (?,?,?)", 1, 100, 0.0) == 1


def test_the_screen_says_names_are_illustrative() -> None:
    """★ Ten san pham la tu dat: man hinh phai noi ro, khong chi ghi trong tai lieu."""
    source = (ROOT / "app" / "shop" / "app.js").read_text(encoding="utf-8")
    assert "là minh họa vì dữ liệu gốc đã ẩn danh" in source and "Mã gốc RetailRocket" in source
    assert "không phải xác suất mua" in source and "Số này khác cách đo ở Chương 4" in source


def test_stylesheet_keeps_the_design_floor() -> None:
    """Quy tac nen lay tu hallmark, kiem bang tay vi khong chay script cua repo ngoai."""
    css = (ROOT / "app" / "shop" / "app.css").read_text(encoding="utf-8")
    body = css[css.index("*{box-sizing"):]
    assert "transition:all" not in css.replace(" ", "") and "overflow-x:hidden" not in css
    assert "overflow-x:clip" in css and "prefers-reduced-motion" in css and ":focus-visible" in css
    assert not re.search(r"text-transform:\s*uppercase", css) and "linear-gradient" not in css
    assert not re.search(r"(?<![\w-])oklch\(", body), "moi mau phai qua token o :root"
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", body)


def test_write_limiter_blocks_one_address_only() -> None:
    from src.shop.limits import WriteLimiter

    clock = [100.0]
    limiter = WriteLimiter(2, clock=lambda: clock[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    assert limiter.allow("b")
    clock[0] += 61
    assert limiter.allow("a")


def test_the_public_page_does_not_carry_the_api_key() -> None:
    assert "demo-key" not in (ROOT / "app" / "shop" / "app.js").read_text(encoding="utf-8")

