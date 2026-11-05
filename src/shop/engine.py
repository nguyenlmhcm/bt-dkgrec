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
from src.graph.schema import NodeSpace
from src.serving.online import BEHAVIORS, edge_weights_at
from src.serving.service import DemoService, event_weight
from src.serving.store import GraphStore
from src.shop.catalog import persona_name, stable_hash
from src.shop.db import ShopDB
from src.shop.paths import PathGraph, strongest_paths, visitor_coefficients

COHORT, MODEL = "active", "bt_dkgrec_l05"
BEHAVIOR_VN = {"view": "xem", "addtocart": "thêm vào giỏ", "transaction": "mua"}
REPLAY_K = 20
NEW_ITEM_BASE = 900_000_000


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def weighted_timeline(history: pd.DataFrame, tau: int, bundle) -> pd.DataFrame:
    """Add each event's age and weight at time ``tau``, by the same formula as the model."""
    out = history.copy()
    out["age_days"] = (tau - out["timestamp"].astype("int64")) / bundle.d_day
    out["weight_now"] = [event_weight(bundle, b, a) for b, a in zip(out["behavior"], out["age_days"])]
    return out


class ShopEngine:
    """Long-lived state of the shop process."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.base_frame = pd.read_parquet(root / "data" / "app" / "catalog.parquet").assign(in_model=True)
        self.db = ShopDB(root / "data" / "app" / "shop.sqlite")
        self.apply_catalog_edits()
        versions = self.model_versions_on_disk()
        if not versions:
            raise FileNotFoundError("Chua co model trong experiments/serving/")
        chosen = self.db.one("SELECT value FROM settings WHERE key='active_model'")
        names = {v.name for v in versions}
        self.model_dir = versions[-1] if not chosen or chosen["value"] not in names else self.root / "experiments" / "serving" / chosen["value"]
        self._load_model(self.model_dir)
        self.store = GraphStore()
        self.visitors = pd.read_csv(root / "data" / "neo4j" / f"{COHORT}_{MODEL}_evaluated" / "visitors.csv")
        self.visitor_ids = self.visitors["visitor_id"].astype(int).tolist()
        self._load_side_names()
        self._load_activity()
        self.replay_state: dict = {"status": "idle"}
        self._replay_lock = threading.Lock()

    # ── model versions ─────────────────────────────────────────────────
    def model_versions_on_disk(self) -> list[Path]:
        files = (self.root / "experiments" / "serving").glob(f"{COHORT}_{MODEL}_*/e0_best.npy")
        return [f.parent for f in sorted(files, key=lambda f: f.stat().st_mtime)]

    def _load_model(self, model_dir: Path) -> None:
        """Build everything that depends on one trained model, then switch in one step."""
        e0_path = model_dir / "e0_best.npy"
        service = DemoService.from_artifacts(COHORT, MODEL, e0_path)
        previous = getattr(self, "service", None)
        self.service = service
        try:
            self._item_row = {int(v): i for i, v in enumerate(service.bundle.item_ids)}
            self._visitor_z, self.paths = self._load_graph(e0_path)
            norm = np.linalg.norm(service.bundle.item_z, axis=1, keepdims=True)
            self._item_unit = service.bundle.item_z / np.maximum(norm, 1e-9)
            self._interest = None
            self.model_meta = json.loads((model_dir / "e0_best.json").read_text(encoding="utf-8"))
            self.model_dir = model_dir
        except Exception:
            self.service = previous
            raise

    def model_versions(self) -> list[dict]:
        out = []
        for path in self.model_versions_on_disk():
            test = json.loads((path / "metrics.json").read_text(encoding="utf-8"))["test"]["warm"] if (path / "metrics.json").exists() else {}
            valid = json.loads((path / "e0_best.json").read_text(encoding="utf-8"))
            out.append({"name": path.name, "trained": (path / "e0_best.npy").stat().st_mtime, "valid_ndcg20": valid["value"],
                        "recall20": test.get("recall@20"), "ndcg20": test.get("ndcg@20"), "active": path == self.model_dir})
        return out

    def apply_model(self, name: str) -> dict:
        target = self.root / "experiments" / "serving" / name
        if target not in self.model_versions_on_disk():
            raise FileNotFoundError(name)
        self._load_model(target)
        self.db.run("INSERT INTO settings(key, value) VALUES ('active_model', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", name)
        return {"active": name}

    def pending_for_training(self) -> dict:
        """New behaviours and catalogue changes the next training would have to take in."""
        logged = self.db.one("SELECT COUNT(*) n FROM event_log WHERE source != 'replay'")["n"]
        new = self.db.one("SELECT COUNT(*) n FROM product_edits WHERE is_new=1 AND deleted=0")["n"]
        return {"new_events": logged, "new_products": new}

    # ── catalogue edits ────────────────────────────────────────────────
    def apply_catalog_edits(self) -> None:
        """Lay the shop's edits over the base catalogue. The trained model does not change."""
        frame = self.base_frame.copy()
        renames = {r["original"]: r["name"] for r in self.db.rows("SELECT * FROM department_edits WHERE original IS NOT NULL")}
        self.dept_original = {new: old for old, new in renames.items()}
        frame["department"] = frame["department"].replace(renames)
        self.extra_departments = [r["name"] for r in self.db.rows("SELECT name FROM department_edits WHERE original IS NULL")]
        known = set(frame["item_id"])
        rows, drop = [], set()
        icon_of = frame.drop_duplicates("department").set_index("department")["icon"].to_dict()
        dept_of = frame.drop_duplicates("department").set_index("department")["dept_idx"].to_dict()
        for e in self.db.rows("SELECT * FROM product_edits"):
            if e["deleted"]:
                drop.add(e["item_id"])
            elif e["is_new"]:
                rows.append({"item_id": e["item_id"], "name": e["name"], "variant": "", "tone": stable_hash("tone", e["item_id"]) % 8,
                             "department": e["department"], "dept_idx": dept_of.get(e["department"], 1000),
                             "icon": icon_of.get(e["department"], "package"), "product_type": e["name"], "category_id": -1, "price": e["price"],
                             "price_is_real": False, "available": bool(e["available"]), "n_view": 0, "n_cart": 0, "n_txn": 0, "in_model": False})
            elif e["item_id"] in known:
                at = frame["item_id"] == e["item_id"]
                updates = {"name": e["name"], "product_type": e["name"], "variant": "", "department": e["department"],
                           "dept_idx": dept_of.get(e["department"], 1000), "icon": icon_of.get(e["department"], "package"),
                           "price": e["price"], "available": bool(e["available"])}
                for column, value in updates.items():
                    frame.loc[at, column] = value
        frame = frame[~frame["item_id"].isin(drop)]
        if rows:
            frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
        self.frame = frame
        self.items = {int(r["item_id"]): r for r in frame.to_dict("records")}
        self.popular = (frame.assign(score=frame.n_txn * 3 + frame.n_cart * 2 + frame.n_view * 0.01)
                        .query("available and in_model").sort_values(["score", "item_id"], ascending=[False, True])["item_id"]
                        .astype(int).tolist()[:200])
        self._interest = None

    def save_product(self, item_id: int | None, name: str, department: str, price: int, available: bool) -> dict:
        is_new = item_id is None or item_id not in set(self.base_frame["item_id"])
        if item_id is None:
            top = self.db.one("SELECT MAX(item_id) m FROM product_edits WHERE is_new=1")["m"]
            item_id = max(top or 0, NEW_ITEM_BASE) + 1
        self.db.run("INSERT INTO product_edits(item_id, name, department, price, available, deleted, is_new) VALUES (?,?,?,?,?,0,?) "
                    "ON CONFLICT(item_id) DO UPDATE SET name=excluded.name, department=excluded.department, price=excluded.price, "
                    "available=excluded.available, deleted=0", item_id, name, department, price, int(available), int(is_new))
        self.apply_catalog_edits()
        return self.items[item_id]

    def delete_product(self, item_id: int) -> None:
        item = self.items[item_id]
        self.db.run("INSERT INTO product_edits(item_id, name, department, price, available, deleted, is_new) VALUES (?,?,?,?,?,1,?) "
                    "ON CONFLICT(item_id) DO UPDATE SET deleted=1",
                    item_id, item["name"], item["department"], int(item["price"]), int(item["available"]), int(not item["in_model"]))
        self.apply_catalog_edits()

    def add_department(self, name: str) -> None:
        self.db.run("INSERT OR IGNORE INTO department_edits(name, original) VALUES (?, NULL)", name)
        self.apply_catalog_edits()

    def rename_department(self, old: str, new: str) -> None:
        if old in self.extra_departments:
            self.db.run("UPDATE department_edits SET name=? WHERE name=?", new, old)
        else:
            original = self.dept_original.get(old, old)
            self.db.run("DELETE FROM department_edits WHERE original=?", original)
            self.db.run("INSERT INTO department_edits(name, original) VALUES (?,?)", new, original)
        self.db.run("UPDATE product_edits SET department=? WHERE department=?", new, old)
        self.apply_catalog_edits()

    def delete_department(self, name: str) -> None:
        if (self.frame["department"] == name).any():
            raise ValueError("Ngành hàng còn sản phẩm, hãy chuyển hoặc xóa sản phẩm trước")
        self.db.run("DELETE FROM department_edits WHERE name=?", name)
        self.apply_catalog_edits()

    # ── customers, for the shop ────────────────────────────────────────
    def customers_overview(self) -> list[dict]:
        events = self.store.all_events()
        if events.empty:
            return []
        price = self.frame.set_index("item_id")["price"]
        events["value"] = events["item_id"].map(price).fillna(0)
        table = events.groupby("visitor_id").agg(n_events=("item_id", "size"), n_items=("item_id", "nunique"), last=("timestamp", "max")).reset_index()
        for behavior, column in (("addtocart", "n_cart"), ("transaction", "n_txn")):
            table[column] = table["visitor_id"].map(events[events["behavior"] == behavior].groupby("visitor_id").size()).fillna(0).astype(int)
        table["spent"] = table["visitor_id"].map(events[events["behavior"] == "transaction"].groupby("visitor_id")["value"].sum()).fillna(0).astype(int)
        table["name"] = table["visitor_id"].map(lambda v: persona_name(int(v)))
        return table.sort_values("last", ascending=False).to_dict("records")

    def customer_detail(self, visitor_id: int) -> dict:
        history = self.store.history(visitor_id)
        rows = []
        for row in history.sort_values("timestamp", ascending=False).head(300).itertuples():
            item = self.items.get(int(row.item_id))
            rows.append({"timestamp": int(row.timestamp), "behavior": BEHAVIOR_VN[row.behavior], "item_id": int(row.item_id),
                         "item": item["name"] if item else f"Sản phẩm {row.item_id}", "new": bool(row.demo)})
        suggestions = [i for i in self.recommend(visitor_id, k=30)["items"] if i["available"]][:6]
        return {"visitor_id": visitor_id, "name": persona_name(visitor_id), "n_events": len(history), "events": rows, "suggestions": suggestions}

    def past_purchases(self, visitor_id: int) -> list[dict]:
        """Purchases already in the behaviour history, for the customer's own page."""
        history = self.store.history(visitor_id)
        if history.empty:
            return []
        bought = history[history["behavior"] == "transaction"].sort_values("timestamp", ascending=False)
        out = []
        for row in bought.itertuples():
            item = self.items.get(int(row.item_id))
            if item and not row.demo:
                out.append({"timestamp": int(row.timestamp), "item": item})
        return out

    def _load_graph(self, e0_path: Path) -> tuple[np.ndarray, PathGraph]:
        """Visitor embeddings of the trained graph, and that graph for reading paths."""
        e0 = np.load(e0_path)
        graph = self.root / "data" / "processed" / COHORT / MODEL
        hat = symmetric_normalize(sp.load_npz(graph / "adjacency.npz")).tocsr()
        layer, total = e0, e0.copy()
        for _ in range(self.service.bundle.num_layers):
            layer = np.asarray(hat @ layer, dtype="float32")
            total = total + layer
        stats = json.loads((graph / "graph_stats.json").read_text(encoding="utf-8"))
        space = NodeSpace(stats["n_visitor"], stats["n_item"], stats["n_category"], stats["n_property_value"])
        z = (total / (self.service.bundle.num_layers + 1))[: stats["n_visitor"]]
        return z, PathGraph(hat, space, self.service.bundle.item_degree)

    def _load_side_names(self) -> None:
        interim = self.root / "data" / "interim" / COHORT
        categories = pd.read_parquet(interim / "side_categories.parquet")
        self._category_id = dict(zip(categories["idx"], categories["category_id"]))
        values = pd.read_parquet(interim / "side_property_values.parquet")
        self._property_key = dict(zip(values["pv_idx"], values["prop_key"]))
        type_of = self.base_frame.drop_duplicates("category_id").set_index("category_id")["product_type"]
        self._category_label = type_of.to_dict()

    def _load_activity(self) -> None:
        """Train events by week and department, and who acted on what, for the shop screens."""
        events = pd.read_parquet(self.root / "data" / "interim" / COHORT / "events.parquet")
        self._future = events[events["split"] != "train"].sort_values("timestamp")
        train = events[events["split"] == "train"][["itemid", "behavior", "timestamp", "visitorid"]].copy()
        department = self.base_frame.set_index("item_id")["department"]
        train["department"] = train["itemid"].map(department).astype("string").fillna("")
        train["week"] = pd.to_datetime(train["timestamp"], unit="ms").dt.to_period("W-SUN").dt.start_time
        grouped = train.groupby(["department", "week", "behavior"], observed=True).size().reset_index(name="n")
        self._weekly = grouped
        target = train[train["behavior"] != "view"].drop_duplicates(["visitorid", "itemid"])
        users = target["visitorid"].astype("category")
        items = target["itemid"].astype("category")
        self._co_items = items.cat.categories.to_numpy()
        self._co_row = {int(v): i for i, v in enumerate(self._co_items)}
        self._co = sp.csr_matrix((np.ones(len(target), dtype="int32"), (users.cat.codes, items.cat.codes)),
                                 shape=(len(users.cat.categories), len(items.cat.categories)))

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
        return frame.sort_values(["in_model", "available", "score"], ascending=[True, False, False]).head(limit).drop(columns="score").to_dict("records")

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

    def record(self, visitor_id: int, item_id: int, behavior: str, source: str = "shop", k: int = 20,
               timestamp: int | None = None) -> dict:
        """The designed flow: write the event, read history back, recompute Top-K."""
        if behavior not in BEHAVIORS:
            raise ValueError(behavior)
        started = time.perf_counter()
        history = self.store.history(visitor_id)
        timestamp = self.service.next_timestamp(history) if timestamp is None else int(timestamp)
        weight = event_weight(self.service.bundle, behavior)
        self.store.ingest(visitor_id, item_id, behavior, timestamp, weight)
        write_ms = (time.perf_counter() - started) * 1000
        result = self.recommend(visitor_id, k=k)
        total = write_ms + result["read_ms"] + result["model_ms"]
        self.db.run("INSERT INTO event_log(created, visitor, item, behavior, source, total_ms) VALUES (?,?,?,?,?,?)",
                    time.time(), visitor_id, item_id, behavior, source, total)
        result.update(write_ms=round(write_ms, 1), total_ms=round(total, 1), weight=weight, behavior=behavior,
                      recorded_at=time.time())
        return result

    # ── graph, as the model sees it ────────────────────────────────────
    def _state(self, visitor_id: int) -> dict:
        """Edge weights and first-hop coefficients of a visitor at their current time."""
        history = self.store.history(visitor_id)
        known = history[history["item_id"].isin(self._item_row)].copy()
        known["item_idx"] = known["item_id"].map(self._item_row).astype("int64")
        tau = max(self.service.t_train, int(known["timestamp"].max()) if len(known) else 0)
        weights = edge_weights_at(known, tau, self.service.bundle) if len(known) else pd.Series(dtype="float64")
        row = self.service._visitor_row.get(int(visitor_id))
        in_graph = None
        if row is not None and len(known):
            in_graph = edge_weights_at(known[known["timestamp"] <= self.service.t_train], self.service.t_train, self.service.bundle)
        return {"history": history, "known": known, "tau": tau, "weights": weights, "row": row,
                "coefficients": visitor_coefficients(weights, in_graph, self.service.bundle.item_degree)}

    def _describe_node(self, node: int) -> dict:
        kind, space = self.paths.kind(node), self.paths.space
        if kind == "customer":
            visitor = int(self.service.visitor_ids[node])
            return {"id": f"c{node}", "kind": kind, "label": persona_name(visitor), "detail": f"Khách {visitor}"}
        if kind == "item":
            item = self.items[int(self.service.bundle.item_ids[node - space.n_visitor])]
            return {"id": f"i{item['item_id']}", "kind": kind, "label": item["name"], "detail": f"Mã {item['item_id']}", "item_id": item["item_id"]}
        if kind == "category":
            category = int(self._category_id[node - space.n_visitor - space.n_item])
            return {"id": f"k{category}", "kind": kind, "label": f"Nhóm {self._category_label.get(category, category)}", "detail": f"Danh mục {category}"}
        key = self._property_key[node - space.n_visitor - space.n_item - space.n_category]
        return {"id": f"p{node}", "kind": kind, "label": f"Thuộc tính {key}", "detail": "Giá trị đã ẩn danh trong dữ liệu gốc"}

    def paths_to(self, visitor_id: int, item_id: int, state: dict | None = None) -> list[dict]:
        """Strongest propagation paths from a visitor's history to one item."""
        state = state or self._state(visitor_id)
        target = self._item_row.get(int(item_id))
        if target is None or state["coefficients"].empty:
            return []
        out = strongest_paths(self.paths, state["coefficients"], target, state["row"])
        offset = self.paths.space.n_visitor
        for path in out:
            path["via"] = self._describe_node(offset + path.pop("item_idx"))
            path["middle"] = self._describe_node(path["middle"])
        return out

    def why(self, visitor_id: int, item_id: int) -> dict:
        """Plain-language reason for the customer screen, from the strongest path."""
        state = self._state(visitor_id)
        item_id = int(item_id)
        known = state["history"]
        if len(known) and item_id in set(known["item_id"].astype(int)):
            return {"kind": "direct"}
        paths = self.paths_to(visitor_id, item_id, state)
        if not paths:
            return {"kind": "none"}
        best = paths[0]
        return {"kind": best["middle"]["kind"], "seen": self.items[best["via"]["item_id"]]["name"],
                "middle": best["middle"]["label"]}

    def also_liked(self, visitor_id: int, skip: set[int], k: int = 6) -> list[dict]:
        """Suggestions whose strongest path runs through another customer."""
        state = self._state(visitor_id)
        out = []
        for item in self.recommend(visitor_id, k=60)["items"]:
            if item["item_id"] in skip or not item["available"]:
                continue
            paths = self.paths_to(visitor_id, item["item_id"], state)
            if paths and paths[0]["middle"]["kind"] == "customer":
                out.append(item)
            if len(out) == k:
                break
        return out

    def bought_with(self, item_id: int, k: int = 6) -> list[dict]:
        """Items most often carted or bought by the same customers, from train events."""
        col = self._co_row.get(int(item_id))
        if col is None:
            return []
        buyers = self._co[:, col].nonzero()[0]
        if len(buyers) == 0:
            return []
        counts = np.asarray(self._co[buyers].sum(axis=0)).ravel()
        counts[col] = 0
        out = []
        for i in np.argsort(-counts)[:40]:
            other = self.items.get(int(self._co_items[i]))
            if counts[i] > 0 and other and other["available"]:
                out.append({**other, "shared_customers": int(counts[i])})
            if len(out) == k:
                break
        return out

    def graph_view(self, visitor_id: int, k: int = 8) -> dict:
        """Everything the provider's graph screen draws for one customer."""
        state = self._state(visitor_id)
        recommended = self.recommend(visitor_id, k=k)
        history = state["history"]
        bundle = self.service.bundle
        events = []
        if len(history):
            age = (state["tau"] - history["timestamp"].astype("int64")) / bundle.d_day
            for row, days in zip(history.sort_values("timestamp", ascending=False).head(40).itertuples(),
                                 age.loc[history.sort_values("timestamp", ascending=False).head(40).index]):
                item = self.item(row.item_id) or {"name": f"Sản phẩm {row.item_id}"}
                events.append({"timestamp": int(row.timestamp), "behavior": BEHAVIOR_VN[row.behavior], "item_id": int(row.item_id),
                               "item": item["name"], "age_days": round(float(days), 1), "alpha": float(bundle.alpha[BEHAVIORS.index(row.behavior)]),
                               "w": round(event_weight(bundle, row.behavior, float(days)), 4), "new": bool(row.demo)})
        edges = []
        for idx, weight in state["weights"].sort_values(ascending=False).items():
            item = self.items.get(int(bundle.item_ids[idx]))
            if item:
                edges.append({"item_id": item["item_id"], "name": item["name"], "W": round(float(weight), 4)})
        return {"visitor_id": visitor_id, "name": persona_name(visitor_id), "tau": state["tau"], "t_train": self.service.t_train,
                "lambda": bundle.lambda_decay, "events": events, "edges": edges, "n_events": len(history),
                "source": recommended["source"], "recommended": recommended["items"]}

    def whole_graph(self) -> dict:
        """Counts of the whole training graph and of the live store."""
        graph = self.root / "data" / "processed" / COHORT / MODEL
        stats = json.loads((graph / "graph_stats.json").read_text(encoding="utf-8"))
        records, _, _ = self.store._driver.execute_query("MATCH (n) RETURN labels(n)[0] AS l, count(*) AS n")
        rels, _, _ = self.store._driver.execute_query("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n")
        return {"training": {k: stats[k] for k in ("n_visitor", "n_item", "n_category", "n_property_value", "n_nodes_total", "edges", "n_edges_undirected", "t_train", "lambda_decay", "alpha")},
                "store_nodes": {r["l"]: r["n"] for r in records}, "store_edges": {r["t"]: r["n"] for r in rels}}

    def next_real_event(self, visitor_id: int) -> dict:
        """Feed the visitor's next real event after the training cut through the designed flow."""
        history = self.store.history(visitor_id)
        after = max(self.service.t_train, int(history["timestamp"].max()) if len(history) else 0)
        future = self._future[(self._future["visitorid"] == visitor_id) & (self._future["timestamp"] > after)
                              & self._future["itemid"].isin(self._item_row)]
        if future.empty:
            return {"done": True}
        event = future.iloc[0]
        item_id = int(event["itemid"])
        before = [i["item_id"] for i in self.recommend(visitor_id, k=20)["items"]]
        touched = set(history["item_id"].astype(int)) if len(history) else set()
        result = self.record(visitor_id, item_id, event["behavior"], source="next", timestamp=int(event["timestamp"]))
        after_ids = [i["item_id"] for i in result["items"]]
        return {"done": False, "timestamp": int(event["timestamp"]), "behavior": BEHAVIOR_VN[event["behavior"]], "item": self.items[item_id],
                "first_touch": item_id not in touched, "rank_before": before.index(item_id) + 1 if item_id in before else None,
                "n_new": len(set(after_ids) - set(before)), "remaining": len(future) - 1, "weight": result["weight"]}

    # ── shop: departments, products and interest ───────────────────────
    def departments_overview(self) -> list[dict]:
        frame = self.frame.groupby("department", as_index=False).agg(
            n=("item_id", "size"), n_view=("n_view", "sum"), n_cart=("n_cart", "sum"), n_txn=("n_txn", "sum"),
            icon=("icon", "first"), out=("available", lambda a: int((~a).sum())))
        rows = frame.sort_values("n_view", ascending=False).to_dict("records")
        have = {r["department"] for r in rows}
        return rows + [{"department": d, "n": 0, "n_view": 0, "n_cart": 0, "n_txn": 0, "icon": "package", "out": 0}
                       for d in self.extra_departments if d not in have]

    def department_detail(self, department: str, limit: int = 15) -> dict:
        items = self.search(department=department, limit=limit)
        weekly = self._weekly[self._weekly["department"] == self.dept_original.get(department, department)]
        table = weekly.pivot_table(index="week", columns="behavior", values="n", aggfunc="sum", fill_value=0).sort_index()
        return {"department": department, "items": items,
                "weeks": [{"week": str(w.date()), "view": int(r.get("view", 0)), "cart": int(r.get("addtocart", 0)), "buy": int(r.get("transaction", 0))}
                          for w, r in table.iterrows()]}

    def interest(self, limit: int = 20) -> list[dict]:
        """How often each product sits in a customer's Top-20, over customers in the store."""
        if self._interest is None:
            mapping = {int(v): i for i, v in enumerate(self.service.visitor_ids)}
            rows = [mapping[v] for v in self.visitor_ids if v in mapping]
            scores = self._visitor_z[rows] @ self.service.bundle.item_z.T.astype("float32")
            top = np.argpartition(-scores, 20, axis=1)[:, :20]
            counts = np.bincount(top.ravel(), minlength=scores.shape[1])
            out = []
            for i in np.argsort(-counts)[:300]:
                item = self.items.get(int(self.service.bundle.item_ids[i]))
                if item and counts[i] > 0:
                    out.append({**item, "n_customers": int(counts[i])})
            self._interest = out
        return self._interest[:limit]

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
                state.update(done=state["done"] + 1, event_time=int(row.timestamp), targets=first_touches, hits=hits, hits_popular=hits_popular,
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
