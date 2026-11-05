"""HTTP layer of the demo shop: public API v1, shop routes and the seller console (D39)."""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.serving.online import BEHAVIORS
from src.shop.catalog import SHOWCASE, persona_name
from src.shop.engine import BEHAVIOR_VN, ShopEngine, percentile
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
def get_recommendations(visitor_id: int, k: int = 20) -> dict:
    """Danh sách gợi ý hiện tại; khách chưa có hành vi nhận danh sách bán chạy."""
    return engine().recommend(visitor_id, k=min(max(k, 1), 100))


@app.get("/api/v1/recommendations/{visitor_id}/why/{item_id}", tags=["API tích hợp"])
def get_why(visitor_id: int, item_id: int) -> dict:
    """Đường nối giữa sản phẩm được gợi ý và lịch sử của khách trong đồ thị."""
    _item_or_404(item_id)
    return engine().why(visitor_id, item_id)


@app.get("/api/v1/items/{item_id}/audience", tags=["API tích hợp"])
def get_audience(item_id: int, k: int = 15) -> list[dict]:
    """Khách được dự báo quan tâm nhất tới một sản phẩm."""
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


@app.get("/shop/personas", include_in_schema=False)
def personas() -> list[dict]:
    e = engine()
    out = []
    for visitor in SHOWCASE:
        history = e.store.history(visitor)
        out.append({"visitor_id": visitor, "name": persona_name(visitor), "n_events": len(history),
                    "n_bought": int((history["behavior"] == "transaction").sum()) if len(history) else 0})
    return out


@app.post("/shop/guest", include_in_schema=False)
def guest() -> dict:
    return {"visitor_id": random.randint(90_000_000, 99_999_999), "name": "Khách mới"}


@app.get("/shop/me/{visitor_id}", include_in_schema=False)
def me(visitor_id: int) -> dict:
    e = engine()
    history = e.store.history(visitor_id)
    timeline = []
    for row in history.sort_values("timestamp", ascending=False).head(30).itertuples():
        item = e.item(row.item_id)
        timeline.append({"behavior": row.behavior, "behavior_vn": BEHAVIOR_VN[row.behavior], "weight": round(float(row.w), 3),
                         "demo": bool(row.demo), "item": item or {"item_id": int(row.item_id), "name": f"Sản phẩm {row.item_id}"}})
    orders = e.db.rows("SELECT * FROM orders WHERE visitor=? ORDER BY id DESC", visitor_id)
    for order in orders:
        order["lines"] = [{**line, "item": e.item(line["item"])} for line in
                          e.db.rows("SELECT * FROM order_items WHERE order_id=?", order["id"])]
    known = visitor_id in e.visitor_ids
    return {"visitor_id": visitor_id, "name": persona_name(visitor_id) if known else "Khách mới", "known": known,
            "n_events": len(history), "n_new": int(history["demo"].sum()) if len(history) else 0,
            "timeline": timeline, "orders": orders, "offers": _offers(visitor_id)}


def _offers(visitor_id: int) -> list[dict]:
    e = engine()
    rows = e.db.rows(
        "SELECT c.id, c.item, c.discount FROM campaigns c JOIN campaign_targets t ON t.campaign_id=c.id "
        "WHERE t.visitor=? AND t.bought IS NULL ORDER BY c.id DESC", visitor_id)
    if rows:
        e.db.run("UPDATE campaign_targets SET seen=COALESCE(seen, ?) WHERE visitor=? AND bought IS NULL", time.time(), visitor_id)
    return [{"campaign_id": r["id"], "discount": r["discount"], "item": e.item(r["item"]),
             "price_after": _discounted(e.item(r["item"])["price"], r["discount"])} for r in rows]


def _discounted(price: int, discount: int) -> int:
    return int(round(price * (100 - discount) / 100 / 1000)) * 1000


@app.get("/shop/items", include_in_schema=False)
def items(q: str = "", department: str = "", limit: int = 40) -> list[dict]:
    return engine().search(q, department, min(limit, 100))


@app.get("/shop/departments", include_in_schema=False)
def departments() -> list[dict]:
    frame = engine().frame
    counts = frame.groupby(["department", "dept_idx", "icon"], as_index=False).size().sort_values("size", ascending=False)
    return counts.rename(columns={"size": "n"}).to_dict("records")


@app.get("/shop/items/{item_id}", include_in_schema=False)
def item_detail(item_id: int) -> dict:
    return {**_item_or_404(item_id), "similar": engine().similar(item_id)}


@app.post("/shop/visit", include_in_schema=False)
def visit(body: VisitIn) -> dict:
    """Opening a product page is a `view` event, as it would be in a real shop."""
    _item_or_404(body.item_id)
    e = engine()
    if body.campaign_id:
        e.db.run("UPDATE campaign_targets SET clicked=COALESCE(clicked, ?) WHERE campaign_id=? AND visitor=?",
                 time.time(), body.campaign_id, body.visitor_id)
    return e.record(body.visitor_id, body.item_id, "view")


@app.get("/shop/cart/{visitor_id}", include_in_schema=False)
def cart(visitor_id: int) -> dict:
    e = engine()
    offers = {o["item"]["item_id"]: o for o in _offers(visitor_id)}
    lines, total = [], 0
    for row in e.db.rows("SELECT item, qty FROM cart WHERE visitor=?", visitor_id):
        item = e.item(row["item"])
        offer = offers.get(row["item"])
        price = offer["price_after"] if offer else item["price"]
        total += price * row["qty"]
        lines.append({"item": item, "qty": row["qty"], "price": price, "discount": offer["discount"] if offer else 0})
    return {"lines": lines, "total": total, "count": sum(line["qty"] for line in lines)}


@app.post("/shop/cart", include_in_schema=False)
def add_to_cart(body: CartIn) -> dict:
    item = _item_or_404(body.item_id)
    if not item["available"]:
        raise HTTPException(409, "Sản phẩm tạm hết hàng")
    e = engine()
    e.db.run("INSERT INTO cart(visitor, item, qty) VALUES (?,?,?) "
             "ON CONFLICT(visitor, item) DO UPDATE SET qty=qty+excluded.qty", body.visitor_id, body.item_id, max(body.qty, 1))
    return {**e.record(body.visitor_id, body.item_id, "addtocart"), "cart": cart(body.visitor_id)}


@app.delete("/shop/cart/{visitor_id}/{item_id}", include_in_schema=False)
def remove_from_cart(visitor_id: int, item_id: int) -> dict:
    engine().db.run("DELETE FROM cart WHERE visitor=? AND item=?", visitor_id, item_id)
    return cart(visitor_id)


@app.post("/shop/checkout", include_in_schema=False)
def checkout(body: CheckoutIn) -> dict:
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
def forget_me(visitor_id: int) -> dict:
    """Customer-side deletion of everything recorded in this shop."""
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


@app.get("/seller/overview", include_in_schema=False)
def overview() -> dict:
    e = engine()
    shop_log = e.db.rows("SELECT total_ms FROM event_log WHERE source != 'replay'")
    latencies = [r["total_ms"] for r in shop_log]
    money = e.db.one("SELECT COUNT(*) n, COALESCE(SUM(total), 0) total FROM orders")
    return {"health": health(), "n_events": len(shop_log), "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1), "n_orders": money["n"], "revenue": money["total"]}


@app.get("/seller/events", include_in_schema=False)
def event_feed(limit: int = 25) -> list[dict]:
    e = engine()
    rows = e.db.rows("SELECT * FROM event_log WHERE source != 'replay' ORDER BY id DESC LIMIT ?", min(limit, 100))
    for row in rows:
        row["name"] = persona_name(row["visitor"]) if row["visitor"] in e.visitor_ids else "Khách mới"
        row["item_name"] = (e.item(row["item"]) or {}).get("name", f"Sản phẩm {row['item']}")
        row["behavior_vn"] = BEHAVIOR_VN[row["behavior"]]
    return rows


@app.get("/seller/abandoned", include_in_schema=False)
def abandoned() -> list[dict]:
    return engine().abandoned_carts()


@app.post("/seller/campaigns", include_in_schema=False)
def create_campaign(body: CampaignIn) -> dict:
    _item_or_404(body.item_id)
    if not body.visitors:
        raise HTTPException(422, "Chọn ít nhất một khách")
    e = engine()
    campaign_id = e.db.run("INSERT INTO campaigns(item, discount, created) VALUES (?,?,?)", body.item_id, body.discount, time.time())
    for visitor in body.visitors:
        e.db.run("INSERT OR IGNORE INTO campaign_targets(campaign_id, visitor) VALUES (?,?)", campaign_id, visitor)
    return {"campaign_id": campaign_id, "n": len(body.visitors)}


@app.get("/seller/campaigns", include_in_schema=False)
def campaigns() -> list[dict]:
    e = engine()
    rows = e.db.rows(
        "SELECT c.*, COUNT(t.visitor) n, COUNT(t.seen) seen, COUNT(t.clicked) clicked, COUNT(t.bought) bought "
        "FROM campaigns c LEFT JOIN campaign_targets t ON t.campaign_id=c.id GROUP BY c.id ORDER BY c.id DESC")
    for row in rows:
        row["item"] = e.item(row["item"])
    return rows


@app.post("/seller/replay", include_in_schema=False)
def start_replay(limit: int = 300) -> dict:
    return engine().start_replay(min(max(limit, 10), 2000))


@app.get("/seller/replay", include_in_schema=False)
def replay_status() -> dict:
    return engine().replay_state


@app.get("/seller/results", include_in_schema=False)
def results() -> dict:
    e = engine()
    serving = json.loads((e.model_dir / "metrics.json").read_text(encoding="utf-8"))["test"]["warm"]
    return {"rows": e.official_results(), "serving": serving}


@app.post("/seller/reset", include_in_schema=False)
def reset_demo() -> dict:
    """Return the shop and the graph to the state of the original import."""
    e = engine()
    removed = sum(e.store.reset(visitor) for visitor in e.store.demo_visitors())
    e.db.clear()
    return {"removed": removed}


app.mount("/assets", StaticFiles(directory=STATIC), name="assets")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
