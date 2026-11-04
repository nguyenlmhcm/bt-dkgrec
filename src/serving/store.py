"""Kho do thi Neo4j cua demo (D38): tiep nhan su kien, cap nhat do thi, doc lich su.

Su kien do demo tao mang co ``demo: true`` de hoan tac duoc ma khong dong vao du lieu goc.
"""

from __future__ import annotations

import pandas as pd
from neo4j import GraphDatabase

COUNTER = {"view": "n_view", "addtocart": "n_cart", "transaction": "n_txn"}


class GraphStore:
    """Connection to the graph store. Every write is one transaction."""

    def __init__(self, uri: str = "bolt://127.0.0.1:7687") -> None:
        self._driver = GraphDatabase.driver(uri, auth=None)

    def close(self) -> None:
        self._driver.close()

    def ping(self) -> int:
        """Number of Event nodes; raises if the store is not reachable."""
        records, _, _ = self._driver.execute_query("MATCH (e:Event) RETURN count(e) AS n")
        return int(records[0]["n"])

    def history(self, visitor_id: int) -> pd.DataFrame:
        """All events of a visitor, oldest first."""
        records, _, _ = self._driver.execute_query(
            "MATCH (:Visitor {visitor_id: $v})-[:PERFORMED]->(e:Event)-[:TARGETS]->(i:Item) "
            "RETURN i.item_id AS item_id, e.behavior AS behavior, e.timestamp AS timestamp, "
            "e.w_event AS w, coalesce(e.demo, false) AS demo ORDER BY e.timestamp",
            v=int(visitor_id))
        return pd.DataFrame([r.data() for r in records],
                            columns=["item_id", "behavior", "timestamp", "w", "demo"])

    def ingest(self, visitor_id: int, item_id: int, behavior: str, timestamp: int, weight: float) -> None:
        """Record one event and add its weight to the visitor-item edge."""
        counter = COUNTER[behavior]     # fixed names only: never user input
        self._driver.execute_query(
            "MATCH (v:Visitor {visitor_id: $v}) "
            "MERGE (i:Item {item_id: $i}) "
            "CREATE (e:Event {event_id: $eid, behavior: $b, timestamp: $t, w_event: $w, demo: true}) "
            "CREATE (v)-[:PERFORMED]->(e) CREATE (e)-[:TARGETS]->(i) "
            "MERGE (v)-[r:INTERACTED_WITH]->(i) "
            "ON CREATE SET r.weight = 0.0, r.n_view = 0, r.n_cart = 0, r.n_txn = 0, r.demo_created = true "
            f"SET r.weight = r.weight + $w, r.{counter} = r.{counter} + 1, r.last_ts = $t",
            v=int(visitor_id), i=int(item_id), b=behavior, t=int(timestamp), w=float(weight),
            eid=f"demo_{visitor_id}_{item_id}_{timestamp}")

    def reset(self, visitor_id: int) -> int:
        """Undo every demo event of a visitor; return how many were removed."""
        records, _, _ = self._driver.execute_query(
            "MATCH (v:Visitor {visitor_id: $v})-[:PERFORMED]->(e:Event {demo: true})-[:TARGETS]->(i:Item) "
            "MATCH (v)-[r:INTERACTED_WITH]->(i) "
            "WITH r, collect(e) AS es, sum(e.w_event) AS w "
            "SET r.weight = r.weight - w, "
            "r.n_view = r.n_view - size([x IN es WHERE x.behavior = 'view']), "
            "r.n_cart = r.n_cart - size([x IN es WHERE x.behavior = 'addtocart']), "
            "r.n_txn = r.n_txn - size([x IN es WHERE x.behavior = 'transaction']) "
            "WITH r, es, size(es) AS n FOREACH (x IN es | DETACH DELETE x) "
            "WITH r, n CALL { WITH r WITH r WHERE r.demo_created = true DELETE r } "
            "RETURN sum(n) AS n", v=int(visitor_id))
        return int(records[0]["n"] or 0)
