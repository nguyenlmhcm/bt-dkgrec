"""Business logic of the demo shop on top of the real serving stack (D39).

Every personalised list comes from `DemoService` (fold-in on the trained model) and every
behaviour goes through `GraphStore` into Neo4j. Nothing is scripted for the demo.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.graph.normalize import symmetric_normalize
from src.serving.online import BEHAVIORS
from src.serving.service import DemoService, event_weight
from src.serving.store import GraphStore
from src.shop.catalog import persona_name
from src.shop.db import ShopDB

COHORT, MODEL = "active", "bt_dkgrec_l05"
BEHAVIOR_VN = {"view": "xem", "addtocart": "thêm vào giỏ", "transaction": "mua"}
REPLAY_K = 20


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def explain(history: pd.DataFrame, item: dict, catalog: dict[int, dict]) -> dict:
    """Connect a suggestion to the visitor's history through the category graph.

    Reads shared categories after ranking, as `src/export/trace.py` does; it describes
    a path in the graph and does not attribute the model's score.
    """
    weight_of = {"view": 1, "addtocart": 2, "transaction": 3}
    best, kind = None, "collaborative"
    for row in history.sort_values("timestamp", ascending=False).itertuples():
        seen = catalog.get(int(row.item_id))
        if seen is None:
            continue
        if int(row.item_id) == item["item_id"]:
            return {"kind": "direct", "behavior": row.behavior, "seen": seen}
        if item["category_id"] != -1 and seen["category_id"] == item["category_id"]:
            if kind != "category" or weight_of[row.behavior] > weight_of[best[1]]:
                best, kind = (seen, row.behavior), "category"
        elif kind == "collaborative" and seen["dept_idx"] == item["dept_idx"] and item["dept_idx"] != -1:
            best, kind = (seen, row.behavior), "department"
        elif kind == "department" and seen["dept_idx"] == item["dept_idx"] and weight_of[row.behavior] > weight_of[best[1]]:
            best = (seen, row.behavior)
    if best is None:
        return {"kind": "collaborative"}
    return {"kind": kind, "seen": best[0], "behavior": best[1]}


class ShopEngine:
    """Long-lived state of the shop process."""

    def __init__(self, root: Path) -> None:
        self.root = root
        frame = pd.read_parquet(root / "data" / "app" / "catalog.parquet")
        self.items: dict[int, dict] = {int(r["item_id"]): r for r in frame.to_dict("records")}
        self.frame = frame
        self.popular = (frame.assign(score=frame.n_txn * 3 + frame.n_cart * 2 + frame.n_view * 0.01)
                        .query("available").sort_values(["score", "item_id"], ascending=[False, True])["item_id"]
                        .astype(int).tolist()[:200])
        files = sorted((root / "experiments" / "serving").glob(f"{COHORT}_{MODEL}_*/e0_best.npy"),
                       key=lambda p: p.stat().st_mtime)
        if not files:
            raise FileNotFoundError("Chua co model trong experiments/serving/")
        self.model_dir = files[-1].parent
        self.service = DemoService.from_artifacts(COHORT, MODEL, files[-1])
        self.model_meta = json.loads((self.model_dir / "e0_best.json").read_text(encoding="utf-8"))
        self.store = GraphStore()
        self.db = ShopDB(root / "data" / "app" / "shop.sqlite")
        self.visitors = pd.read_csv(root / "data" / "neo4j" / f"{COHORT}_{MODEL}_evaluated" / "visitors.csv")
        self.visitor_ids = self.visitors["visitor_id"].astype(int).tolist()
        self._item_row = {int(v): i for i, v in enumerate(self.service.bundle.item_ids)}
        self._visitor_z = self._propagate_visitors(files[-1])
        norm = np.linalg.norm(self.service.bundle.item_z, axis=1, keepdims=True)
        self._item_unit = self.service.bundle.item_z / np.maximum(norm, 1e-9)
        self.replay_state: dict = {"status": "idle"}
        self._replay_lock = threading.Lock()

    def _propagate_visitors(self, e0_path: Path) -> np.ndarray:
        """Final visitor embeddings for the seller's audience list (trained graph)."""
        e0 = np.load(e0_path)
        graph = self.root / "data" / "processed" / COHORT / MODEL
        hat = symmetric_normalize(sp.load_npz(graph / "adjacency.npz"))
        layer, total = e0, e0.copy()
        for _ in range(self.service.bundle.num_layers):
            layer = np.asarray(hat @ layer, dtype="float32")
            total = total + layer
        n_visitor = json.loads((graph / "graph_stats.json").read_text(encoding="utf-8"))["n_visitor"]
        return (total / (self.service.bundle.num_layers + 1))[:n_visitor]

    # ── items ──────────────────────────────────────────────────────────
    def item(self, item_id: int) -> dict | None:
        return self.items.get(int(item_id))

    def similar(self, item_id: int, k: int = 8) -> list[dict]:
        """Nearest items by learnt embedding."""
        row = self._item_row.get(int(item_id))
        if row is None:
            return []
        scores = self._item_unit @ self._item_unit[row]
        scores[row] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [self.items[int(self.service.bundle.item_ids[i])] for i in top
                if int(self.service.bundle.item_ids[i]) in self.items]

    def search(self, q: str = "", department: str = "", limit: int = 40) -> list[dict]:
        frame = self.frame
        if department:
            frame = frame[frame["department"] == department]
        if q:
            frame = frame[frame["name"].str.contains(q, case=False, regex=False) | (frame["item_id"].astype(str) == q)]
        frame = frame.assign(score=frame.n_txn * 3 + frame.n_cart * 2 + frame.n_view * 0.01)
        return frame.sort_values(["available", "score"], ascending=False).head(limit).drop(columns="score").to_dict("records")

    # ── recommendations ────────────────────────────────────────────────
    def recommend(self, visitor_id: int, k: int = 20) -> dict:
        """Top-K from the visitor's current history in Neo4j."""
        started = time.perf_counter()
        history = self.store.history(visitor_id)
        read_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        known = history[history["item_id"].isin(self._item_row)] if len(history) else history
        if len(known) == 0:
            ids, source = self.popular[:k], "popular"
        else:
            top = self.service.recommend(visitor_id, history, k=k)
            ids, source = top["item_id"].astype(int).tolist(), "personal"
        model_ms = (time.perf_counter() - started) * 1000
        return {"visitor_id": visitor_id, "source": source, "items": [self.items[i] for i in ids if i in self.items],
                "n_events": len(history), "read_ms": round(read_ms, 1), "model_ms": round(model_ms, 1)}

    def record(self, visitor_id: int, item_id: int, behavior: str, source: str = "shop", k: int = 20) -> dict:
        """The designed flow: write the event, read history back, recompute Top-K."""
        if behavior not in BEHAVIORS:
            raise ValueError(behavior)
        started = time.perf_counter()
        history = self.store.history(visitor_id)
        timestamp = self.service.next_timestamp(history)
        weight = event_weight(self.service.bundle, behavior)
        self.store.ingest(visitor_id, item_id, behavior, timestamp, weight)
        write_ms = (time.perf_counter() - started) * 1000
        result = self.recommend(visitor_id, k=k)
        total = write_ms + result["read_ms"] + result["model_ms"]
        self.db.run("INSERT INTO event_log(created, visitor, item, behavior, source, total_ms) VALUES (?,?,?,?,?,?)",
                    time.time(), visitor_id, item_id, behavior, source, total)
        result.update(write_ms=round(write_ms, 1), total_ms=round(total, 1), weight=weight, behavior=behavior)
        return result

    def why(self, visitor_id: int, item_id: int) -> dict:
        return explain(self.store.history(visitor_id), self.items[int(item_id)], self.items)

    # ── seller ─────────────────────────────────────────────────────────
    def audience(self, item_id: int, k: int = 15) -> list[dict]:
        """Customers in the store ranked by the model's score for one product."""
        row = self._item_row[int(item_id)]
        mapping = {int(v): i for i, v in enumerate(self.service.visitor_ids)}
        ids = np.array([v for v in self.visitor_ids if v in mapping])
        scores = self._visitor_z[[mapping[int(v)] for v in ids]] @ self.service.bundle.item_z[row]
        events = self.store.all_events()
        item = self.items[int(item_id)]
        events["dept"] = events["item_id"].map(lambda i: self.items.get(int(i), {}).get("dept_idx", -2))
        bought = set(events[(events.item_id == item_id) & (events.behavior == "transaction")].visitor_id)
        same = events[events["dept"] == item["dept_idx"]].groupby("visitor_id").size().to_dict()
        target = events[events.behavior != "view"].groupby("visitor_id").size().to_dict()
        order = np.argsort(-scores)
        out = []
        for i in order:
            visitor = int(ids[i])
            if visitor in bought:
                continue
            out.append({"visitor_id": visitor, "name": persona_name(visitor), "score": round(float(scores[i]), 3),
                        "rank": len(out) + 1, "same_department": int(same.get(visitor, 0)),
                        "intent_events": int(target.get(visitor, 0))})
            if len(out) == k:
                break
        return out

    def abandoned_carts(self, limit: int = 12) -> list[dict]:
        """Items a customer put in the cart and has not bought since, newest first."""
        events = self.store.all_events()
        carts = events[events.behavior == "addtocart"].sort_values("timestamp").drop_duplicates(["visitor_id", "item_id"], keep="last")
        bought = events[events.behavior == "transaction"].groupby(["visitor_id", "item_id"])["timestamp"].max().to_dict()
        rows = [r for r in carts.itertuples() if bought.get((r.visitor_id, r.item_id), -1) < r.timestamp and int(r.item_id) in self.items]
        rows.sort(key=lambda r: -r.timestamp)
        shown: dict[int, int] = {}
        out = []
        for r in rows:
            visitor = int(r.visitor_id)
            if shown.get(visitor, 0) < 2:    # keep one heavy shopper from filling the list
                shown[visitor] = shown.get(visitor, 0) + 1
                out.append({"visitor_id": visitor, "name": persona_name(visitor), "item": self.items[int(r.item_id)]})
            if len(out) == limit:
                break
        return out

    def official_results(self) -> list[dict]:
        """Mean test metrics over three seeds, read from the official runs (thesis Chapter 4)."""
        rows: dict[str, list[dict]] = {}
        for path in sorted((self.root / "experiments" / "runs").glob("active_*/metrics.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.setdefault(data["model"], []).append(data["test"]["warm"])
        keep = [("popularity", "Bán chạy (không cá nhân hóa)"), ("lightgcn", "LightGCN"),
                ("static_kg_gcn", "Đồ thị tri thức tĩnh"), ("bt_dkgrec_l05", "BT-DKGRec-GCN (đề xuất)")]
        out = []
        for key, label in keep:
            if key in rows:
                out.append({"model": key, "label": label, "n_seeds": len(rows[key]),
                            **{m: float(np.mean([r[m] for r in rows[key]]))
                               for m in ("recall@20", "ndcg@20", "hit_rate@20", "coverage@20")}})
        return out

    # ── replay of the real test period ─────────────────────────────────
    def replay_events(self, limit: int) -> pd.DataFrame:
        events = pd.read_parquet(self.root / "data" / "interim" / COHORT / "events.parquet")
        events = events[(events["split"] == "test") & events["visitorid"].isin(self.visitor_ids)
                        & events["itemid"].isin(self.items)]
        return events.sort_values("timestamp").head(limit)

    def start_replay(self, limit: int = 300) -> dict:
        with self._replay_lock:
            if self.replay_state.get("status") == "running":
                return self.replay_state
            self.replay_state = {"status": "running", "done": 0, "total": limit, "started": time.time()}
        threading.Thread(target=self._run_replay, args=(limit,), daemon=True).start()
        return self.replay_state

    def _run_replay(self, limit: int) -> None:
        """Push real test-period events through the serving path, in time order.

        Top-K leaves out items the visitor already touched, so a cart event right after a
        view can never hit. The fair question is the first touch: when a visitor reaches an
        item for the first time, was it already on their list? The same check runs against
        the bestseller list. Written events are removed afterwards.
        """
        state = self.replay_state
        latencies: list[float] = []
        hits = hits_popular = first_touches = 0
        popular = set(self.popular[:REPLAY_K])
        current: dict[int, set[int]] = {}
        touched: dict[int, set[int]] = {}
        try:
            events = self.replay_events(limit)
            state["total"] = len(events)
            for row in events.itertuples():
                visitor, item = int(row.visitorid), int(row.itemid)
                if visitor not in current:
                    start = self.recommend(visitor, k=REPLAY_K)
                    current[visitor] = {i["item_id"] for i in start["items"]}
                    touched[visitor] = set(self.store.history(visitor)["item_id"].astype(int))
                if item not in touched[visitor]:
                    first_touches += 1
                    hits += item in current[visitor]
                    hits_popular += item in popular
                    touched[visitor].add(item)
                result = self.record(visitor, item, row.behavior, source="replay", k=REPLAY_K)
                current[visitor] = {i["item_id"] for i in result["items"]}
                latencies.append(result["total_ms"])
                state.update(done=state["done"] + 1, targets=first_touches, hits=hits, hits_popular=hits_popular,
                             p50=round(percentile(latencies, 50), 1), p95=round(percentile(latencies, 95), 1),
                             visitors=len(current))
            state["status"] = "done"
        except Exception as error:  # noqa: BLE001 - reported to the console screen
            state.update(status="error", error=str(error))
        finally:
            state["seconds"] = round(time.time() - state["started"], 1)
            for visitor in current:
                self.store.reset(visitor)
            state["cleaned"] = True
