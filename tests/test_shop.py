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


def _toy_graph():
    """Visitor 0; items 1, 2, 3; category 4. Item 1 and 2 share a customer (5) and item 1, 3 a category."""
    import numpy as np
    import scipy.sparse as sp

    from src.graph.normalize import symmetric_normalize, weighted_degree
    from src.graph.schema import NodeSpace
    from src.shop.paths import PathGraph

    # order: visitors 0, 1; items 2, 3, 4; category 5
    edges = [(0, 2, 2.0), (1, 2, 1.0), (1, 3, 3.0), (2, 5, 1.0), (4, 5, 1.0)]
    row, col, val = zip(*edges)
    a = sp.coo_matrix((val + val, (row + col, col + row)), shape=(6, 6)).tocsr()
    space = NodeSpace(2, 3, 1, 0)
    return a, PathGraph(symmetric_normalize(a).tocsr(), space, np.asarray(weighted_degree(a))[2:5].astype("float64"))


def test_first_hop_equals_the_normalised_matrix_when_nothing_changed() -> None:
    """★ Khong co hanh vi moi thi he so buoc dau phai trung dung A_hat cua mo hinh."""
    from src.shop.paths import visitor_coefficients

    _, graph = _toy_graph()
    weights = pd.Series({0: 2.0})
    coefficients = visitor_coefficients(weights, weights, graph.item_degree)
    assert coefficients[0] == pytest.approx(graph.a_hat[0, 2])


def test_paths_run_through_the_right_middle_node_and_multiply_edges() -> None:
    from src.shop.paths import strongest_paths, visitor_coefficients

    _, graph = _toy_graph()
    weights = pd.Series({0: 2.0})
    coefficients = visitor_coefficients(weights, weights, graph.item_degree)
    via_customer = strongest_paths(graph, coefficients, 1, 0)
    assert [(p["kind"], p["middle"]) for p in via_customer] == [("customer", 1)]
    expected = graph.a_hat[0, 2] * graph.a_hat[2, 1] * graph.a_hat[1, 3]
    assert via_customer[0]["strength"] == pytest.approx(expected)
    assert [p["kind"] for p in strongest_paths(graph, coefficients, 2, 0)] == ["category"]


def test_the_visitor_is_never_a_middle_node_of_their_own_path() -> None:
    from src.shop.paths import strongest_paths, visitor_coefficients

    _, graph = _toy_graph()
    weights = pd.Series({0: 2.0})
    paths = strongest_paths(graph, visitor_coefficients(weights, weights, graph.item_degree), 0, 0)
    assert all(p["middle"] != 0 for p in paths)


def test_cart_records_survive_a_clear_only_when_asked(tmp_path: Path) -> None:
    from src.shop.db import ShopDB

    db = ShopDB(tmp_path / "s.sqlite")
    order = db.run("INSERT INTO orders(visitor, total, created) VALUES (?,?,?)", 1, 100, 0.0)
    assert order == 1 and db.one("SELECT total FROM orders")["total"] == 100
    db.clear()
    assert db.rows("SELECT * FROM orders") == []
    assert db.run("INSERT INTO orders(visitor, total, created) VALUES (?,?,?)", 1, 100, 0.0) == 1


def _js() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted((ROOT / "app" / "shop").glob("*.js")))


def test_the_screen_says_names_are_illustrative() -> None:
    """★ Ten san pham la tu dat: man hinh phai noi ro, khong chi ghi trong tai lieu."""
    source = _js()
    assert "là minh họa vì dữ liệu gốc đã ẩn danh" in source and "Mã gốc RetailRocket" in source
    assert "không phải xác suất mua" in source and "khác cách đo ở Chương 4" in source
    assert "không phải phân rã điểm" in source and "không phải dự báo số lượng" in source


def test_the_customer_screen_has_no_model_vocabulary() -> None:
    """★ Tach ba cong: nguoi mua khong thay thuat ngu mo hinh."""
    source = (ROOT / "app" / "shop" / "customer.js").read_text(encoding="utf-8")
    for word in ("Neo4j", "Trọng số", "embedding", "Top-K", "Mã gốc", "RetailRocket"):
        assert word not in source, word


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
    assert "demo-key" not in _js()


def _history(*rows: tuple[int, str]) -> pd.DataFrame:
    return pd.DataFrame({"item_id": [r[0] for r in rows], "behavior": [r[1] for r in rows],
                         "timestamp": list(range(len(rows)))})


def test_timeline_weight_follows_the_model_formula() -> None:
    """★ Trong so tren man hinh phai la cong thuc cua mo hinh, khong phai so ve rieng."""
    from types import SimpleNamespace

    import numpy as np

    from src.shop.engine import weighted_timeline

    bundle = SimpleNamespace(alpha=np.array([1.0, 2.0, 3.0]), lambda_decay=0.05, d_day=86_400_000)
    tau = 40 * 86_400_000
    out = weighted_timeline(_history((1, "addtocart")).assign(timestamp=[10 * 86_400_000]), tau, bundle)
    assert out["age_days"].iloc[0] == 30
    assert out["weight_now"].iloc[0] == pytest.approx(2 * np.exp(-0.05 * 30))


def test_the_screens_no_longer_show_latency() -> None:
    source = _js()
    assert "độ trễ" not in source.lower() and "thời gian xử lý" not in source.lower() and "_ms" not in source
    assert "Thời gian ghi nhận" in source


def test_password_is_hashed_salted_and_checked(tmp_path: Path) -> None:
    from src.shop.auth import Accounts, hash_password, verify_password
    from src.shop.db import ShopDB

    assert hash_password("x") != hash_password("x") and verify_password("x", hash_password("x"))
    accounts = Accounts(ShopDB(tmp_path / "s.sqlite"))
    accounts.upsert("a", "dung", "store", "A")
    assert "dung" not in str(accounts.db.rows("SELECT * FROM users"))
    assert accounts.login("a", "sai") is None
    token, user = accounts.login("a", "dung")
    assert user["role"] == "store" and accounts.user_for(token)["username"] == "a"
    accounts.logout(token)
    assert accounts.user_for(token) is None


def test_login_locks_after_repeated_failures_and_sessions_expire(tmp_path: Path) -> None:
    from src.shop.auth import SESSION_SECONDS, Accounts
    from src.shop.db import ShopDB

    now = [1000.0]
    accounts = Accounts(ShopDB(tmp_path / "s.sqlite"), clock=lambda: now[0])
    accounts.upsert("a", "dung", "store", "A")
    token, _ = accounts.login("a", "dung")
    for _ in range(5):
        accounts.login("a", "sai")
    assert accounts.login("a", "dung") is None
    now[0] += SESSION_SECONDS + 1
    assert accounts.user_for(token) is None
    assert accounts.login("a", "dung") is not None


@pytest.fixture
def gate_client(tmp_path: Path, monkeypatch):
    """The HTTP layer with accounts only: gates must answer before any model code runs."""
    pytest.importorskip("httpx")
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    import src.shop.app as shop_app
    from src.shop.auth import Accounts
    from src.shop.db import ShopDB

    db = ShopDB(tmp_path / "s.sqlite")
    engine = SimpleNamespace(db=db, accounts=Accounts(db), item=lambda i: None, items={}, visitor_ids=[1, 2])
    engine.accounts.upsert("k1", "pw", "customer", "K1", 1)
    engine.accounts.upsert("ch", "pw", "store", "CH")
    engine.accounts.upsert("ncc", "pw", "provider", "NCC")
    monkeypatch.setattr(shop_app, "_engine", engine)
    return TestClient(shop_app.app)


def _login(client, username: str, role: str):
    response = client.post("/auth/login", json={"username": username, "password": "pw", "role": role})
    assert response.status_code == 200
    return client


def test_every_console_route_refuses_without_login(gate_client) -> None:
    """★ Trang cong khai: moi duong quan tri phai bi chan khi chua dang nhap."""
    from src.shop.app import app

    checked = 0
    for route in app.routes:
        path = getattr(route, "path", "")
        if path.startswith(("/seller", "/owner")):
            for method in route.methods - {"HEAD", "OPTIONS"}:
                url = path.replace("{visitor_id}", "1").replace("{item_id}", "1").replace("{department}", "x").replace("{name}", "x")
                assert gate_client.request(method, url, json={}).status_code == 401, f"{method} {path}"
                checked += 1
    assert checked > 15


def test_roles_do_not_open_each_others_portal(gate_client) -> None:
    _login(gate_client, "k1", "customer")
    assert gate_client.get("/seller/customers").status_code == 401
    assert gate_client.get("/owner/models").status_code == 401
    _login(gate_client, "ch", "store")
    assert gate_client.get("/owner/models").status_code == 401


def test_an_account_cannot_sign_in_through_another_portal(gate_client) -> None:
    response = gate_client.post("/auth/login", json={"username": "k1", "password": "pw", "role": "provider"})
    assert response.status_code == 401


def test_a_customer_reads_only_their_own_data(gate_client) -> None:
    _login(gate_client, "k1", "customer")
    assert gate_client.get("/shop/me/2").status_code == 403
    assert gate_client.get("/shop/cart/2").status_code == 403
    assert gate_client.delete("/shop/me/2").status_code == 403
    assert gate_client.post("/shop/checkout", json={"visitor_id": 2}).status_code == 403
    assert gate_client.get("/api/v1/recommendations/2").status_code == 403


def test_a_catalogue_edit_changes_the_shop_not_the_model(tmp_path: Path) -> None:
    """★ Sua danh muc chi la lop phu: san pham moi chua co trong mo hinh va khong duoc goi y."""
    from src.shop.db import ShopDB
    from src.shop.engine import ShopEngine

    engine = ShopEngine.__new__(ShopEngine)
    engine.db = ShopDB(tmp_path / "s.sqlite")
    engine.base_frame = pd.DataFrame([
        {"item_id": 1, "name": "A", "variant": "v", "tone": 0, "department": "D1", "dept_idx": 0, "icon": "x", "product_type": "A", "category_id": 5,
         "price": 1000, "price_is_real": True, "available": True, "n_view": 1, "n_cart": 1, "n_txn": 1, "in_model": True},
        {"item_id": 2, "name": "B", "variant": "v", "tone": 0, "department": "D1", "dept_idx": 0, "icon": "x", "product_type": "B", "category_id": 5,
         "price": 2000, "price_is_real": True, "available": True, "n_view": 1, "n_cart": 0, "n_txn": 0, "in_model": True}])
    engine.apply_catalog_edits()
    created = engine.save_product(None, "Moi", "D1", 5000, True)
    assert created["in_model"] is False and created["item_id"] not in engine.popular
    engine.save_product(1, "A sua", "D1", 1500, True)
    assert engine.items[1]["name"] == "A sua" and engine.items[1]["price"] == 1500
    engine.delete_product(2)
    assert 2 not in engine.items and 2 not in engine.popular
    with pytest.raises(ValueError):
        engine.delete_department("D1")
    engine.rename_department("D1", "D moi")
    assert engine.items[1]["department"] == "D moi" and engine.items[created["item_id"]]["department"] == "D moi"
    engine.db.clear()
    engine.apply_catalog_edits()
    assert set(engine.items) == {1, 2} and engine.items[1]["name"] == "A"

