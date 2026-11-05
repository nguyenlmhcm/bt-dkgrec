"""HTTP layer of the demo shop: public API v1, shop routes and the seller console (D39)."""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.serving.online import BEHAVIORS
from src.shop.auth import COOKIE, ROLES, Accounts
from src.shop.catalog import persona_name
from src.shop.engine import BEHAVIOR_VN, ShopEngine, weighted_timeline
from src.shop.limits import WriteLimiter

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "app" / "shop"
API_KEY = os.environ.get("SHOP_API_KEY", "demo-key")
_engine: ShopEngine | None = None

@asynccontextmanager
async def _lifespan(_: FastAPI):
    engine()    # load the model at start-up, not on the first customer's request
    yield


app = FastAPI(title="BT-DKGRec API", version="1.0", lifespan=_lifespan,
              description="Thu nhận hành vi và trả danh sách gợi ý theo thời gian gần thực. "
                          "Tên và giá sản phẩm là minh họa; dữ liệu gốc RetailRocket đã ẩn danh.")
_limiter = WriteLimiter(int(os.environ.get("SHOP_WRITES_PER_MINUTE", "120")))


@app.middleware("http")
async def _limit_writes(request: Request, call_next):
    """The site is public: one address must not be able to flood the graph store."""
    if request.method != "GET" and not _limiter.allow(request.client.host if request.client else "?"):
        return JSONResponse({"detail": "Thao tác quá nhanh, thử lại sau ít phút"}, status_code=429)
    return await call_next(request)


def engine() -> ShopEngine:
    global _engine
    if _engine is None:
        _engine = ShopEngine(ROOT)
    return _engine


# ── Accounts ────────────────────────────────────────────────────────────
def accounts() -> Accounts:
    e = engine()
    if not hasattr(e, "accounts"):
        e.accounts = Accounts(e.db)
    return e.accounts


def _session(request: Request, role: str) -> dict | None:
    return accounts().user_for(request.cookies.get(COOKIE[role]))


def _require(request: Request, role: str) -> dict:
    user = _session(request, role)
    if user is None:
        raise HTTPException(401, "Cần đăng nhập")
    return user


def require_store(request: Request) -> dict:
    return _require(request, "store")


def require_provider(request: Request) -> dict:
    return _require(request, "provider")


def require_customer(request: Request) -> dict:
    return _require(request, "customer")


def _own(user: dict, visitor_id: int) -> None:
    if user["visitor_id"] != visitor_id:
        raise HTTPException(403, "Không phải dữ liệu của tài khoản này")


def _may_read_visitor(request: Request, visitor_id: int, x_api_key: str) -> None:
    """A customer reads their own list; the shop, the provider and API clients read any."""
    if x_api_key == API_KEY or _session(request, "store") or _session(request, "provider"):
        return
    user = _session(request, "customer")
    if user is None:
        raise HTTPException(401, "Cần đăng nhập")
    _own(user, visitor_id)


class LoginIn(BaseModel):
    username: str
    password: str
    role: str


@app.post("/auth/login", include_in_schema=False)
def login(body: LoginIn, request: Request, response: Response) -> dict:
    if body.role not in ROLES:
        raise HTTPException(422, "Cổng không hợp lệ")
    result = accounts().login(body.username.strip(), body.password)
    if result is None or result[1]["role"] != body.role:
        raise HTTPException(401, "Sai tài khoản hoặc mật khẩu")
    token, user = result
    response.set_cookie(COOKIE[body.role], token, httponly=True, samesite="lax", max_age=7 * 24 * 3600,
                        secure=request.url.scheme == "https", path="/")
    return user


@app.post("/auth/logout", include_in_schema=False)
def logout(role: str, request: Request, response: Response) -> dict:
    if role in ROLES:
        accounts().logout(request.cookies.get(COOKIE[role]))
        response.delete_cookie(COOKIE[role], path="/")
    return {"ok": True}


@app.get("/auth/me", include_in_schema=False)
def auth_me(role: str, request: Request) -> dict:
    if role not in ROLES:
        raise HTTPException(422, "Cổng không hợp lệ")
    return _require(request, role)


def _item_or_404(item_id: int) -> dict:
    item = engine().item(item_id)
    if item is None:
        raise HTTPException(404, "Không có sản phẩm này")
    return item


def _require_key(x_api_key: str = Header(default="")) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(401, "Thiếu hoặc sai khóa X-API-Key")


class EventIn(BaseModel):
    visitor_id: int = Field(description="Mã khách trong hệ thống của cửa hàng")
    item_id: int = Field(description="Mã sản phẩm")
    behavior: str = Field(description="Một trong: view, addtocart, transaction")


# ── Public API: what another shop integrates against ─────────────────────
@app.post("/api/v1/events", tags=["API tích hợp"], dependencies=[Depends(_require_key)])
def post_event(event: EventIn) -> dict:
    """Ghi một hành vi vào đồ thị và trả danh sách gợi ý đã tính lại."""
    if event.behavior not in BEHAVIORS:
        raise HTTPException(422, f"behavior phải thuộc {BEHAVIORS}")
    _item_or_404(event.item_id)
    return engine().record(event.visitor_id, event.item_id, event.behavior, source="api")


@app.get("/api/v1/recommendations/{visitor_id}", tags=["API tích hợp"])
def get_recommendations(visitor_id: int, request: Request, k: int = 20, x_api_key: str = Header(default="")) -> dict:
    """Danh sách gợi ý hiện tại; khách chưa có hành vi nhận danh sách bán chạy."""
    _may_read_visitor(request, visitor_id, x_api_key)
    return engine().recommend(visitor_id, k=min(max(k, 1), 100))


@app.get("/api/v1/recommendations/{visitor_id}/why/{item_id}", tags=["API tích hợp"])
def get_why(visitor_id: int, item_id: int, request: Request, x_api_key: str = Header(default="")) -> dict:
    """Đường lan truyền mạnh nhất giữa lịch sử của khách và sản phẩm được gợi ý."""
    _may_read_visitor(request, visitor_id, x_api_key)
    _item_or_404(item_id)
    return engine().why(visitor_id, item_id)


@app.get("/api/v1/items/{item_id}/audience", tags=["API tích hợp"])
def get_audience(item_id: int, request: Request, k: int = 15, x_api_key: str = Header(default="")) -> list[dict]:
    """Khách được dự báo quan tâm nhất tới một sản phẩm."""
    if x_api_key != API_KEY and not (_session(request, "store") or _session(request, "provider")):
        raise HTTPException(401, "Cần đăng nhập")
    _item_or_404(item_id)
    return engine().audience(item_id, k=min(max(k, 1), 100))


@app.get("/api/v1/health", tags=["API tích hợp"])
def health() -> dict:
    """Tình trạng kho đồ thị và mô hình đang phục vụ."""
    e = engine()
    try:
        n_events, ok = e.store.ping(), True
    except Exception:  # noqa: BLE001 - health must answer even when Neo4j is down
        n_events, ok = 0, False
    return {"neo4j": ok, "n_events": n_events, "n_items": len(e.items), "n_visitors": len(e.visitor_ids),
            "model": "BT-DKGRec-GCN (λ=0.05)", "valid_ndcg20": e.model_meta["value"], "model_dir": e.model_dir.name}


# ── Shop ────────────────────────────────────────────────────────────────
class CartIn(BaseModel):
    visitor_id: int
    item_id: int
    qty: int = 1
    campaign_id: int | None = None


class VisitIn(BaseModel):
    visitor_id: int
    item_id: int
    campaign_id: int | None = None


class CheckoutIn(BaseModel):
    visitor_id: int


@app.get("/shop/me/{visitor_id}", include_in_schema=False)
def me(visitor_id: int, user: dict = Depends(require_customer)) -> dict:
    _own(user, visitor_id)
    e = engine()
    history = e.store.history(visitor_id)
    tau = max(e.service.t_train, int(history["timestamp"].max()) if len(history) else 0)
    timeline = []
    if len(history):
        rows = weighted_timeline(history, tau, e.service.bundle).sort_values("timestamp", ascending=False).head(60)
        for row in rows.itertuples():
            item = e.item(row.item_id)
            timeline.append({"behavior": row.behavior, "behavior_vn": BEHAVIOR_VN[row.behavior],
                             "timestamp": int(row.timestamp), "age_days": round(float(row.age_days), 1),
                             "weight": round(float(row.weight_now), 3), "demo": bool(row.demo),
                             "item": item or {"item_id": int(row.item_id), "name": f"Sản phẩm {row.item_id}"}})
    orders = e.db.rows("SELECT * FROM orders WHERE visitor=? ORDER BY id DESC", visitor_id)
    for order in orders:
        order["lines"] = [{**line, "item": e.item(line["item"])} for line in
                          e.db.rows("SELECT * FROM order_items WHERE order_id=?", order["id"])]
    known = visitor_id in e.visitor_ids
    return {"visitor_id": visitor_id, "name": persona_name(visitor_id), "known": known,
            "n_events": len(history), "n_new": int(history["demo"].sum()) if len(history) else 0, "model_time": tau,
            "timeline": timeline, "orders": orders, "past_purchases": e.past_purchases(visitor_id), "offers": _offers(visitor_id)}


def _offers(visitor_id: int) -> list[dict]:
    e = engine()
    rows = e.db.rows(
        "SELECT c.id, c.item, c.discount FROM campaigns c JOIN campaign_targets t ON t.campaign_id=c.id "
        "WHERE t.visitor=? AND t.bought IS NULL ORDER BY c.id DESC", visitor_id)
    if rows:
        e.db.run("UPDATE campaign_targets SET seen=COALESCE(seen, ?) WHERE visitor=? AND bought IS NULL", time.time(), visitor_id)
    return [{"campaign_id": r["id"], "discount": r["discount"], "item": e.item(r["item"]),
             "price_after": _discounted(e.item(r["item"])["price"], r["discount"])} for r in rows if e.item(r["item"])]


def _discounted(price: int, discount: int) -> int:
    return int(round(price * (100 - discount) / 100 / 1000)) * 1000


@app.get("/shop/items", include_in_schema=False)
def items(q: str = "", department: str = "", limit: int = 40) -> list[dict]:
    return engine().search(q, department, min(limit, 100))


@app.get("/shop/departments", include_in_schema=False)
def departments() -> list[dict]:
    e = engine()
    counts = e.frame.groupby(["department", "dept_idx", "icon"], as_index=False).size().sort_values("size", ascending=False)
    rows = counts.rename(columns={"size": "n"}).to_dict("records")
    return rows + [{"department": d, "dept_idx": 1000, "icon": "package", "n": 0} for d in e.extra_departments if d not in set(counts["department"])]


@app.get("/shop/items/{item_id}", include_in_schema=False)
def item_detail(item_id: int) -> dict:
    return {**_item_or_404(item_id), "bought_with": engine().bought_with(item_id), "similar": engine().similar(item_id)}


@app.get("/shop/also/{visitor_id}", include_in_schema=False)
def also(visitor_id: int, skip: str = "", user: dict = Depends(require_customer)) -> list[dict]:
    _own(user, visitor_id)
    return engine().also_liked(visitor_id, {int(x) for x in skip.split(",") if x})


@app.post("/shop/visit", include_in_schema=False)
def visit(body: VisitIn, user: dict = Depends(require_customer)) -> dict:
    """Opening a product page is a `view` event, as it would be in a real shop."""
    _own(user, body.visitor_id)
    _item_or_404(body.item_id)
    e = engine()
    if body.campaign_id:
        e.db.run("UPDATE campaign_targets SET clicked=COALESCE(clicked, ?) WHERE campaign_id=? AND visitor=?",
                 time.time(), body.campaign_id, body.visitor_id)
    return e.record(body.visitor_id, body.item_id, "view")


@app.get("/shop/cart/{visitor_id}", include_in_schema=False)
def get_cart(visitor_id: int, user: dict = Depends(require_customer)) -> dict:
    _own(user, visitor_id)
    return cart(visitor_id)


def cart(visitor_id: int) -> dict:
    e = engine()
    offers = {o["item"]["item_id"]: o for o in _offers(visitor_id)}
    lines, total = [], 0
    for row in e.db.rows("SELECT item, qty FROM cart WHERE visitor=?", visitor_id):
        item = e.item(row["item"])
        if item is None:    # the shop removed it after it went into the cart
            continue
        offer = offers.get(row["item"])
        price = offer["price_after"] if offer else item["price"]
        total += price * row["qty"]
        lines.append({"item": item, "qty": row["qty"], "price": price, "discount": offer["discount"] if offer else 0})
    return {"lines": lines, "total": total, "count": sum(line["qty"] for line in lines)}


@app.post("/shop/cart", include_in_schema=False)
def add_to_cart(body: CartIn, user: dict = Depends(require_customer)) -> dict:
    _own(user, body.visitor_id)
    item = _item_or_404(body.item_id)
    if not item["available"]:
        raise HTTPException(409, "Sản phẩm tạm hết hàng")
    e = engine()
    e.db.run("INSERT INTO cart(visitor, item, qty) VALUES (?,?,?) "
             "ON CONFLICT(visitor, item) DO UPDATE SET qty=qty+excluded.qty", body.visitor_id, body.item_id, max(body.qty, 1))
    return {**e.record(body.visitor_id, body.item_id, "addtocart"), "cart": cart(body.visitor_id)}


@app.delete("/shop/cart/{visitor_id}/{item_id}", include_in_schema=False)
def remove_from_cart(visitor_id: int, item_id: int, user: dict = Depends(require_customer)) -> dict:
    _own(user, visitor_id)
    engine().db.run("DELETE FROM cart WHERE visitor=? AND item=?", visitor_id, item_id)
    return cart(visitor_id)


@app.post("/shop/checkout", include_in_schema=False)
def checkout(body: CheckoutIn, user: dict = Depends(require_customer)) -> dict:
    _own(user, body.visitor_id)
    e = engine()
    current = cart(body.visitor_id)
    if not current["lines"]:
        raise HTTPException(409, "Giỏ hàng đang trống")
    offers = {o["item"]["item_id"]: o for o in _offers(body.visitor_id)}
    order_id = e.db.run("INSERT INTO orders(visitor, total, created) VALUES (?,?,?)", body.visitor_id, current["total"], time.time())
    result = {}
    for line in current["lines"]:
        item_id = line["item"]["item_id"]
        offer = offers.get(item_id)
        e.db.run("INSERT INTO order_items VALUES (?,?,?,?,?)", order_id, item_id, line["qty"], line["price"],
                 offer["campaign_id"] if offer else None)
        if offer:
            e.db.run("UPDATE campaign_targets SET bought=? WHERE campaign_id=? AND visitor=?",
                     time.time(), offer["campaign_id"], body.visitor_id)
        result = e.record(body.visitor_id, item_id, "transaction")
    e.db.run("DELETE FROM cart WHERE visitor=?", body.visitor_id)
    return {**result, "order_id": order_id, "total": current["total"]}


@app.delete("/shop/me/{visitor_id}", include_in_schema=False)
def forget_me(visitor_id: int, user: dict = Depends(require_customer)) -> dict:
    """Customer-side deletion of everything recorded in this shop."""
    _own(user, visitor_id)
    e = engine()
    removed = e.store.reset(visitor_id)
    for table, column in (("cart", "visitor"), ("orders", "visitor"), ("event_log", "visitor"), ("campaign_targets", "visitor")):
        e.db.run(f"DELETE FROM {table} WHERE {column}=?", visitor_id)  # noqa: S608 - fixed names only
    return {"removed": removed}


# ── Seller console ──────────────────────────────────────────────────────
class CampaignIn(BaseModel):
    item_id: int
    visitors: list[int]
    discount: int = Field(default=10, ge=1, le=50)


@app.get("/seller/overview", include_in_schema=False, dependencies=[Depends(require_store)])
def overview() -> dict:
    e = engine()
    logged = e.db.one("SELECT COUNT(*) n, MAX(created) last FROM event_log WHERE source != 'replay'")
    money = e.db.one("SELECT COUNT(*) n, COALESCE(SUM(total), 0) total FROM orders")
    return {"health": health(), "n_events": logged["n"], "last_event": logged["last"], "n_orders": money["n"], "revenue": money["total"]}


@app.get("/seller/events", include_in_schema=False, dependencies=[Depends(require_store)])
def event_feed(limit: int = 25) -> list[dict]:
    e = engine()
    rows = e.db.rows("SELECT * FROM event_log WHERE source != 'replay' ORDER BY id DESC LIMIT ?", min(limit, 100))
    for row in rows:
        row["name"] = persona_name(row["visitor"]) if row["visitor"] in e.visitor_ids else "Khách mới"
        row["item_name"] = (e.item(row["item"]) or {}).get("name", f"Sản phẩm {row['item']}")
        row["behavior_vn"] = BEHAVIOR_VN[row["behavior"]]
    return rows


@app.get("/seller/abandoned", include_in_schema=False, dependencies=[Depends(require_store)])
def abandoned() -> list[dict]:
    return engine().abandoned_carts()


@app.post("/seller/campaigns", include_in_schema=False, dependencies=[Depends(require_store)])
def create_campaign(body: CampaignIn) -> dict:
    _item_or_404(body.item_id)
    if not body.visitors:
        raise HTTPException(422, "Chọn ít nhất một khách")
    e = engine()
    campaign_id = e.db.run("INSERT INTO campaigns(item, discount, created) VALUES (?,?,?)", body.item_id, body.discount, time.time())
    for visitor in body.visitors:
        e.db.run("INSERT OR IGNORE INTO campaign_targets(campaign_id, visitor) VALUES (?,?)", campaign_id, visitor)
    return {"campaign_id": campaign_id, "n": len(body.visitors)}


@app.get("/seller/campaigns", include_in_schema=False, dependencies=[Depends(require_store)])
def campaigns() -> list[dict]:
    e = engine()
    rows = e.db.rows(
        "SELECT c.*, COUNT(t.visitor) n, COUNT(t.seen) seen, COUNT(t.clicked) clicked, COUNT(t.bought) bought "
        "FROM campaigns c LEFT JOIN campaign_targets t ON t.campaign_id=c.id GROUP BY c.id ORDER BY c.id DESC")
    for row in rows:
        row["item"] = e.item(row["item"]) or {"name": f"Sản phẩm {row['item']} (đã xóa)"}
    return rows


@app.post("/owner/replay", include_in_schema=False, dependencies=[Depends(require_provider)])
def start_replay(limit: int = 300) -> dict:
    return engine().start_replay(min(max(limit, 10), 2000))


@app.get("/owner/replay", include_in_schema=False, dependencies=[Depends(require_provider)])
def replay_status() -> dict:
    return engine().replay_state


@app.get("/owner/results", include_in_schema=False, dependencies=[Depends(require_provider)])
def results() -> dict:
    e = engine()
    serving = json.loads((e.model_dir / "metrics.json").read_text(encoding="utf-8"))["test"]["warm"]
    return {"rows": e.official_results(), "serving": serving}


@app.get("/seller/departments", include_in_schema=False, dependencies=[Depends(require_store)])
def seller_departments() -> list[dict]:
    return engine().departments_overview()


@app.get("/seller/departments/{department}", include_in_schema=False, dependencies=[Depends(require_store)])
def seller_department(department: str) -> dict:
    return engine().department_detail(department)


@app.get("/seller/interest", include_in_schema=False, dependencies=[Depends(require_store)])
def seller_interest(limit: int = 20) -> list[dict]:
    return engine().interest(min(limit, 100))


# ── Provider console: how the model and the graph behave ─────────────────
@app.get("/owner/customers", include_in_schema=False, dependencies=[Depends(require_provider)])
def owner_customers() -> list[dict]:
    return [{"visitor_id": v, "name": persona_name(v)} for v in engine().visitor_ids]


@app.get("/owner/whole-graph", include_in_schema=False, dependencies=[Depends(require_provider)])
def owner_whole_graph() -> dict:
    return engine().whole_graph()


class ApplyIn(BaseModel):
    name: str


@app.get("/owner/models", include_in_schema=False, dependencies=[Depends(require_provider)])
def owner_models() -> dict:
    e = engine()
    return {"versions": e.model_versions(), "pending": e.pending_for_training()}


@app.post("/owner/models/apply", include_in_schema=False, dependencies=[Depends(require_provider)])
def owner_apply(body: ApplyIn) -> dict:
    try:
        return engine().apply_model(body.name)
    except FileNotFoundError:
        raise HTTPException(404, "Không có phiên bản mô hình này") from None


@app.get("/owner/graph/{visitor_id}", include_in_schema=False, dependencies=[Depends(require_provider)])
def owner_graph(visitor_id: int) -> dict:
    return engine().graph_view(visitor_id)


@app.get("/owner/graph/{visitor_id}/paths/{item_id}", include_in_schema=False, dependencies=[Depends(require_provider)])
def owner_paths(visitor_id: int, item_id: int) -> list[dict]:
    _item_or_404(item_id)
    return engine().paths_to(visitor_id, item_id)


@app.post("/owner/graph/{visitor_id}/next", include_in_schema=False, dependencies=[Depends(require_provider)])
def owner_next(visitor_id: int) -> dict:
    return engine().next_real_event(visitor_id)


@app.post("/owner/graph/{visitor_id}/reset", include_in_schema=False, dependencies=[Depends(require_provider)])
def owner_reset(visitor_id: int) -> dict:
    return {"removed": engine().store.reset(visitor_id)}


# ── Shop: catalogue and customers ───────────────────────────────────────
class ProductIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    department: str = Field(min_length=2, max_length=80)
    price: int = Field(ge=1000, le=100_000_000)
    available: bool = True


class DepartmentIn(BaseModel):
    name: str = Field(min_length=2, max_length=80)


@app.post("/seller/products", include_in_schema=False, dependencies=[Depends(require_store)])
def create_product(body: ProductIn) -> dict:
    return engine().save_product(None, body.name.strip(), body.department, body.price, body.available)


@app.put("/seller/products/{item_id}", include_in_schema=False, dependencies=[Depends(require_store)])
def update_product(item_id: int, body: ProductIn) -> dict:
    _item_or_404(item_id)
    return engine().save_product(item_id, body.name.strip(), body.department, body.price, body.available)


@app.delete("/seller/products/{item_id}", include_in_schema=False, dependencies=[Depends(require_store)])
def remove_product(item_id: int) -> dict:
    _item_or_404(item_id)
    engine().delete_product(item_id)
    return {"ok": True}


@app.post("/seller/department-names", include_in_schema=False, dependencies=[Depends(require_store)])
def create_department(body: DepartmentIn) -> dict:
    engine().add_department(body.name.strip())
    return {"ok": True}


@app.put("/seller/department-names/{name}", include_in_schema=False, dependencies=[Depends(require_store)])
def rename_department(name: str, body: DepartmentIn) -> dict:
    engine().rename_department(name, body.name.strip())
    return {"ok": True}


@app.delete("/seller/department-names/{name}", include_in_schema=False, dependencies=[Depends(require_store)])
def delete_department(name: str) -> dict:
    try:
        engine().delete_department(name)
    except ValueError as error:
        raise HTTPException(409, str(error)) from None
    return {"ok": True}


@app.get("/seller/customers", include_in_schema=False, dependencies=[Depends(require_store)])
def seller_customers() -> list[dict]:
    return engine().customers_overview()


@app.get("/seller/customers/{visitor_id}", include_in_schema=False, dependencies=[Depends(require_store)])
def seller_customer(visitor_id: int) -> dict:
    return engine().customer_detail(visitor_id)


@app.post("/owner/reset", include_in_schema=False, dependencies=[Depends(require_provider)])
def reset_demo() -> dict:
    """Return the shop and the graph to the state of the original import."""
    e = engine()
    removed = sum(e.store.reset(visitor) for visitor in e.store.demo_visitors())
    e.db.clear()
    e.apply_catalog_edits()
    return {"removed": removed}


app.mount("/assets", StaticFiles(directory=STATIC), name="assets")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
