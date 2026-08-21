"""Build the projected training graph from preprocessed artefacts.

Consumes only data that already passed the Buoc 3 guards: train-only, mapped to
matrix indices, and cut at ``T_train``. After building, the guards run again on
the assembled edges, because an index or timestamp defect introduced while
assembling would otherwise reach the model unnoticed.

The graph is undirected and weighted. Interaction edges carry ``W(u,i)`` from
:mod:`src.graph.weighting`; side-information edges carry the per-relation weight
from the config (1.0 in the reported configuration, KG_DESIGN.md muc 3).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data.mapping import IdMapping
from src.graph.normalize import assert_symmetric, symmetric_normalize, weighted_degree
from src.graph.schema import Graph, NodeSpace, RelationType
from src.graph.weighting import EdgeWeighting, aggregate_interaction_edges
from src.guards.leakage import assert_edges_within_train, assert_index_within_mapping
from src.utils.config import Config
from src.utils.logging import get_logger

log = get_logger(__name__)


def load_interim(interim_dir: Path) -> dict[str, object]:
    """Load everything ``01_preprocess.py`` wrote for one cohort."""
    return {
        "events": pd.read_parquet(interim_dir / "events.parquet"),
        "mapping": IdMapping.load(interim_dir),
        "item_category": pd.read_parquet(interim_dir / "side_item_category.parquet"),
        "item_property": pd.read_parquet(interim_dir / "side_item_property.parquet"),
        "property_values": pd.read_parquet(interim_dir / "side_property_values.parquet"),
        "categories": pd.read_parquet(interim_dir / "side_categories.parquet"),
        "category_parent": pd.read_parquet(interim_dir / "side_category_parent.parquet"),
        "split": json.loads((interim_dir / "split.json").read_text(encoding="utf-8")),
    }


def build_interaction_edges(
    events: pd.DataFrame, mapping: IdMapping, weighting: EdgeWeighting, cfg: Config, t_train: int
) -> pd.DataFrame:
    """Aggregate train events into weighted visitor-item edges.

    Raises:
        ValueError: If a train event references an id outside the mapping,
            which would mean the mapping and the events disagree.
    """
    train = events[events["split"] == "train"].copy()
    train["visitor_idx"] = mapping.visitor_index(train["visitorid"])
    train["item_idx"] = mapping.item_index(train["itemid"])

    unmapped = (train["visitor_idx"] < 0) | (train["item_idx"] < 0)
    if unmapped.any():
        raise ValueError(
            f"{int(unmapped.sum()):,} su kien train khong anh xa duoc — mapping va events lech nhau"
        )

    return aggregate_interaction_edges(
        train_events=train,
        weighting=weighting,
        t_train=t_train,
        d_day=cfg.weighting.d_day,
        behaviors=tuple(cfg.data.history_behaviors),
        target_behaviors=tuple(cfg.data.target_behaviors),
    )


def build_graph(cfg: Config, interim_dir: Path, weighting: EdgeWeighting) -> tuple[Graph, pd.DataFrame]:
    """Assemble the symmetric weighted adjacency matrix for one model variant.

    Args:
        cfg: Resolved configuration; ``cfg.model.use_side_info`` decides whether
            Category and PropertyValue blocks exist at all.
        interim_dir: ``data/interim/<cohort>/``.
        weighting: Strategy from :func:`src.graph.weighting.weighting_for_model`.

    Returns:
        The built :class:`Graph` and the aggregated interaction edge table
        (kept for the Neo4j export of Buoc 11).
    """
    data = load_interim(interim_dir)
    mapping: IdMapping = data["mapping"]  # type: ignore[assignment]
    t_train = int(data["split"]["t_train"])  # type: ignore[index]
    use_side_info = cfg.model.use_side_info

    edges = build_interaction_edges(data["events"], mapping, weighting, cfg, t_train)  # type: ignore[arg-type]

    node_space = NodeSpace(
        n_visitor=mapping.n_visitors,
        n_item=mapping.n_items,
        n_category=len(data["categories"]) if use_side_info else 0,  # type: ignore[arg-type]
        n_property_value=len(data["property_values"]) if use_side_info else 0,  # type: ignore[arg-type]
    )

    rows: list[np.ndarray] = [node_space.visitor_ids(edges["visitor_idx"].to_numpy())]
    cols: list[np.ndarray] = [node_space.item_ids(edges["item_idx"].to_numpy())]
    vals: list[np.ndarray] = [edges["weight"].to_numpy().astype("float64")]
    edge_counts = {RelationType.INTERACTED_WITH.value: len(edges)}

    if use_side_info:
        item_category: pd.DataFrame = data["item_category"]  # type: ignore[assignment]
        item_property: pd.DataFrame = data["item_property"]  # type: ignore[assignment]
        category_parent: pd.DataFrame = data["category_parent"]  # type: ignore[assignment]

        rows.append(node_space.item_ids(item_category["item_idx"].to_numpy()))
        cols.append(node_space.category_ids(item_category["category_idx"].to_numpy()))
        vals.append(np.full(len(item_category), cfg.graph.rel_weight_item_category))
        edge_counts[RelationType.HAS_CATEGORY.value] = len(item_category)

        rows.append(node_space.item_ids(item_property["item_idx"].to_numpy()))
        cols.append(node_space.property_value_ids(item_property["pv_idx"].to_numpy()))
        vals.append(np.full(len(item_property), cfg.graph.rel_weight_item_property))
        edge_counts[RelationType.HAS_PROPERTY.value] = len(item_property)

        rows.append(node_space.category_ids(category_parent["category_idx"].to_numpy()))
        cols.append(node_space.category_ids(category_parent["parent_idx"].to_numpy()))
        vals.append(np.full(len(category_parent), cfg.graph.rel_weight_category_parent))
        edge_counts[RelationType.PARENT_CATEGORY.value] = len(category_parent)

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    val = np.concatenate(vals)

    # Store each undirected edge in both directions so A is symmetric.
    size = node_space.total
    adjacency = sp.coo_matrix(
        (np.concatenate([val, val]), (np.concatenate([row, col]), np.concatenate([col, row]))),
        shape=(size, size),
    ).tocsr()
    adjacency.sum_duplicates()

    stats = _graph_stats(cfg, adjacency, node_space, edge_counts, edges, weighting, t_train)
    graph = Graph(
        adjacency=adjacency, node_space=node_space, edge_counts=edge_counts, stats=stats
    )
    run_graph_guards(graph, edges, mapping, t_train, use_side_info, data)
    return graph, edges


def _graph_stats(
    cfg: Config,
    adjacency: sp.csr_matrix,
    node_space: NodeSpace,
    edge_counts: dict[str, int],
    edges: pd.DataFrame,
    weighting: EdgeWeighting,
    t_train: int,
) -> dict[str, object]:
    """Collect the figures written to ``graph_stats.json`` (KG_DESIGN.md muc 8)."""
    degree = weighted_degree(adjacency)
    return {
        "cohort": cfg.cohort.name,
        "model": cfg.model.name,
        "use_side_info": cfg.model.use_side_info,
        **weighting.describe(),
        **node_space.as_dict(),
        "edges": dict(edge_counts),
        "n_edges_undirected": int(sum(edge_counts.values())),
        "n_nonzero_symmetric": int(adjacency.nnz),
        "interaction_weight_min": float(edges["weight"].min()),
        "interaction_weight_max": float(edges["weight"].max()),
        "interaction_weight_mean": float(edges["weight"].mean()),
        "adjacency_value_min": float(adjacency.data.min()),
        "adjacency_value_max": float(adjacency.data.max()),
        "degree_min": float(degree.min()),
        "degree_max": float(degree.max()),
        "isolated_nodes": int((degree == 0).sum()),
        "max_edge_timestamp": int(edges["last_ts"].max()),
        "t_train": t_train,
        "max_events_per_pair": int((edges["n_view"] + edges["n_cart"] + edges["n_txn"]).max()),
    }


def run_graph_guards(
    graph: Graph,
    edges: pd.DataFrame,
    mapping: IdMapping,
    t_train: int,
    use_side_info: bool,
    data: dict[str, object],
) -> None:
    """Verify the assembled graph before it can reach a model.

    Runs the leakage guards on the aggregated edges and the structural checks of
    KG_DESIGN.md muc 8. Any failure raises -- a graph that fails validation is
    never trained on.

    Raises:
        LeakageError: On a leakage rule violation.
        ValueError: On a structural defect.
    """
    stats = graph.stats

    assert_edges_within_train(edges, t_train, name="INTERACTED_WITH", column="last_ts")
    assert_index_within_mapping(edges, "visitor_idx", mapping.n_visitors, "interaction edges")
    assert_index_within_mapping(edges, "item_idx", mapping.n_items, "interaction edges")

    if float(stats["interaction_weight_min"]) <= 0:  # type: ignore[arg-type]
        raise ValueError(f"canh co trong so <= 0: min={stats['interaction_weight_min']}")

    if int(stats["isolated_nodes"]) != 0:  # type: ignore[arg-type]
        raise ValueError(f"con {stats['isolated_nodes']} node bac 0 sau khi dung graph")

    if use_side_info:
        item_category: pd.DataFrame = data["item_category"]  # type: ignore[assignment]
        if len(item_category) > mapping.n_items:
            raise ValueError(
                f"n_edge_category={len(item_category):,} > n_item={mapping.n_items:,}: "
                "moi item chi duoc co toi da mot category tai T_train"
            )

    assert_symmetric(graph.adjacency)
    log.info(
        "guard graph OK: %s node, %s canh vo huong, khong node co lap, ma tran doi xung",
        f"{graph.node_space.total:,}", f"{graph.n_edges:,}",
    )


def save_graph(
    graph: Graph, edges: pd.DataFrame, out_dir: Path, save_normalized: bool = False
) -> None:
    """Persist the graph and its statistics.

    ``A_hat`` is not stored by default: it is a deterministic function of ``A``
    and recomputing it at load time costs seconds while a stale copy on disk
    could silently disagree with the matrix it came from.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "adjacency.npz", graph.adjacency)
    edges.to_parquet(out_dir / "edges_interacted.parquet", index=False)
    (out_dir / "graph_stats.json").write_text(
        json.dumps(graph.stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if save_normalized:
        sp.save_npz(out_dir / "adjacency_normalized.npz", symmetric_normalize(graph.adjacency))
    log.info("da ghi graph vao %s", out_dir)
