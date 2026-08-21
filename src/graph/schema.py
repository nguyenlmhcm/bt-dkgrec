"""Graph schema: node types, relation types, and the unified index space.

Mirrors docs/KG_DESIGN.md muc 2, 3 and 6.1. The training layer projects the four
node types into one contiguous index space so a single sparse matrix carries the
whole graph::

    [0, n_visitor)                        Visitor
    [n_visitor, +n_item)                  Item
    [.., +n_category)                     Category
    [.., +n_property_value)               PropertyValue

The evaluator relies on these offsets to slice the Visitor x Item block, so the
layout is defined once here and never recomputed by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import scipy.sparse as sp


class NodeType(str, Enum):
    """Node types of the projected training layer."""

    VISITOR = "visitor"
    ITEM = "item"
    CATEGORY = "category"
    PROPERTY_VALUE = "property_value"


class RelationType(str, Enum):
    """Edge types carried into the training layer.

    ``Event`` nodes and the ``PERFORMED`` / ``TARGETS`` relations exist only in
    the Neo4j trace layer; keeping them here would inflate the graph by ~2.7M
    nodes without adding a learning signal (KG_DESIGN.md muc 1).
    """

    INTERACTED_WITH = "interacted_with"
    HAS_CATEGORY = "has_category"
    HAS_PROPERTY = "has_property"
    PARENT_CATEGORY = "parent_category"


@dataclass(frozen=True)
class NodeSpace:
    """Offsets of the unified node index space."""

    n_visitor: int
    n_item: int
    n_category: int = 0
    n_property_value: int = 0

    @property
    def visitor_offset(self) -> int:
        return 0

    @property
    def item_offset(self) -> int:
        return self.n_visitor

    @property
    def category_offset(self) -> int:
        return self.n_visitor + self.n_item

    @property
    def property_value_offset(self) -> int:
        return self.n_visitor + self.n_item + self.n_category

    @property
    def total(self) -> int:
        """Total number of nodes in the projected graph."""
        return self.n_visitor + self.n_item + self.n_category + self.n_property_value

    def visitor_ids(self, local: np.ndarray) -> np.ndarray:
        """Map visitor indices into the unified space."""
        return local.astype("int64") + self.visitor_offset

    def item_ids(self, local: np.ndarray) -> np.ndarray:
        """Map item indices into the unified space."""
        return local.astype("int64") + self.item_offset

    def category_ids(self, local: np.ndarray) -> np.ndarray:
        """Map category indices into the unified space."""
        return local.astype("int64") + self.category_offset

    def property_value_ids(self, local: np.ndarray) -> np.ndarray:
        """Map PropertyValue indices into the unified space."""
        return local.astype("int64") + self.property_value_offset

    def item_slice(self) -> slice:
        """Slice selecting the Item block, used when scoring candidates."""
        return slice(self.item_offset, self.item_offset + self.n_item)

    def visitor_slice(self) -> slice:
        """Slice selecting the Visitor block."""
        return slice(self.visitor_offset, self.visitor_offset + self.n_visitor)

    def as_dict(self) -> dict[str, int]:
        return {
            "n_visitor": self.n_visitor,
            "n_item": self.n_item,
            "n_category": self.n_category,
            "n_property_value": self.n_property_value,
            "visitor_offset": self.visitor_offset,
            "item_offset": self.item_offset,
            "category_offset": self.category_offset,
            "property_value_offset": self.property_value_offset,
            "n_nodes_total": self.total,
        }


@dataclass(frozen=True)
class Graph:
    """A built, symmetric, weighted graph ready for normalisation.

    Attributes:
        adjacency: Symmetric CSR matrix ``A`` over the unified node space.
        node_space: Offsets of the four node blocks.
        edge_counts: Number of *undirected* edges per relation type.
        stats: Everything written to ``graph_stats.json``.
    """

    adjacency: sp.csr_matrix
    node_space: NodeSpace
    edge_counts: dict[str, int] = field(default_factory=dict)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def n_edges(self) -> int:
        """Total undirected edges."""
        return sum(self.edge_counts.values())
