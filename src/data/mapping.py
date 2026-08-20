"""Train-only identifier mapping.

Leakage rule 2: visitor and item indices are built **exclusively** from train
events. A visitor or item that first appears in validation or test has no index
and therefore cannot be scored -- which is the honest behaviour for a temporal
forecasting task.

Category and PropertyValue indices are added by :mod:`src.data.side_info`; the
four node types are later concatenated into one contiguous index space
(KG_DESIGN.md muc 6.1), whose offsets are recorded in ``mappings.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class IdMapping:
    """Bidirectional mapping between raw ids and contiguous matrix indices.

    Attributes:
        visitor_ids: Sorted unique train visitor ids; position = index.
        item_ids: Sorted unique train item ids; position = index.
    """

    visitor_ids: np.ndarray
    item_ids: np.ndarray

    @classmethod
    def from_train_events(cls, train_events: pd.DataFrame) -> "IdMapping":
        """Build the mapping from train events only.

        Raises:
            ValueError: If ``train_events`` still contains non-train rows.
        """
        if "split" in train_events.columns and (train_events["split"] != "train").any():
            raise ValueError("IdMapping must be built from train rows only (leakage rule 2)")
        mapping = cls(
            visitor_ids=np.sort(train_events["visitorid"].unique()),
            item_ids=np.sort(train_events["itemid"].unique()),
        )
        log.info(
            "mapping tu train: %s visitors, %s items",
            f"{mapping.n_visitors:,}", f"{mapping.n_items:,}",
        )
        return mapping

    @property
    def n_visitors(self) -> int:
        return int(len(self.visitor_ids))

    @property
    def n_items(self) -> int:
        return int(len(self.item_ids))

    def visitor_index(self, visitor_ids: pd.Series) -> pd.Series:
        """Map raw visitor ids to indices; unknown ids become ``-1``."""
        return _lookup(visitor_ids, self.visitor_ids)

    def item_index(self, item_ids: pd.Series) -> pd.Series:
        """Map raw item ids to indices; unknown ids become ``-1``."""
        return _lookup(item_ids, self.item_ids)

    def save(self, directory: Path) -> None:
        """Write ``visitors.parquet`` and ``items.parquet``."""
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"visitor_id": self.visitor_ids, "idx": np.arange(self.n_visitors, dtype="int32")}
        ).to_parquet(directory / "visitors.parquet", index=False)
        pd.DataFrame(
            {"item_id": self.item_ids, "idx": np.arange(self.n_items, dtype="int32")}
        ).to_parquet(directory / "items.parquet", index=False)

    @classmethod
    def load(cls, directory: Path) -> "IdMapping":
        """Read a mapping written by :meth:`save`."""
        visitors = pd.read_parquet(directory / "visitors.parquet")
        items = pd.read_parquet(directory / "items.parquet")
        return cls(
            visitor_ids=visitors.sort_values("idx")["visitor_id"].to_numpy(),
            item_ids=items.sort_values("idx")["item_id"].to_numpy(),
        )


def _lookup(values: pd.Series, universe: np.ndarray) -> pd.Series:
    """Vectorised index lookup into a sorted id array, ``-1`` when absent."""
    pos = np.searchsorted(universe, values.to_numpy())
    pos = np.clip(pos, 0, len(universe) - 1)
    found = universe[pos] == values.to_numpy()
    return pd.Series(np.where(found, pos, -1).astype("int32"), index=values.index)


def write_node_offsets(
    directory: Path,
    n_visitors: int,
    n_items: int,
    n_categories: int,
    n_property_values: int,
) -> dict[str, dict[str, int]]:
    """Record the unified node index space used by the graph layer.

    The four blocks are laid out contiguously::

        [0, n_visitor)                        Visitor
        [n_visitor, +n_item)                  Item
        [.., +n_category)                     Category
        [.., +n_property_value)               PropertyValue

    The evaluator relies on these offsets to slice the Visitor x Item block.
    """
    offsets = {
        "visitor": {"start": 0, "count": n_visitors},
        "item": {"start": n_visitors, "count": n_items},
        "category": {"start": n_visitors + n_items, "count": n_categories},
        "property_value": {
            "start": n_visitors + n_items + n_categories,
            "count": n_property_values,
        },
    }
    total = n_visitors + n_items + n_categories + n_property_values
    payload = {"offsets": offsets, "n_nodes_total": total}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "mappings.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("node space: %s nodes tong cong", f"{total:,}")
    return offsets
