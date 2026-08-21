"""Common recommender interface.

Every model -- ``popularity``, ``recent_popularity``, ``lightgcn``,
``static_kg_gcn``, ``bt_dkgrec`` -- implements :class:`Recommender`. The
evaluator talks only to this interface and never learns which concrete model it
is holding. That is what makes the comparison fair: identical candidate set,
identical seen-filtering, identical metrics, identical protocol, with the model
as the only moving part (CLAUDE.md muc "Nguyen tac so sanh hop le").

Division of labour, deliberately strict:

* A model produces **scores** for items, nothing else.
* The evaluator owns candidate scope, seen-filtering, top-K and metrics.

A model that computed its own top-K or its own metric could quietly use a
different protocol from its baselines, which is exactly the failure mode the
thesis has to rule out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.mapping import IdMapping
from src.utils.config import Config


@dataclass(frozen=True)
class ModelContext:
    """Everything a model may read while fitting.

    Only train-derived material appears here. Validation and test events are
    deliberately absent so a model cannot see them even by accident.

    Attributes:
        cfg: Resolved configuration.
        mapping: Train-only visitor/item mapping.
        train_events: Train events with ``visitor_idx`` and ``item_idx``.
        interaction_edges: Aggregated ``W(u,i)`` edges from the graph layer.
        t_train: End of the training window, in milliseconds.
    """

    cfg: Config
    mapping: IdMapping
    train_events: pd.DataFrame
    interaction_edges: pd.DataFrame
    t_train: int

    @property
    def n_items(self) -> int:
        return self.mapping.n_items

    @property
    def n_visitors(self) -> int:
        return self.mapping.n_visitors


class Recommender(ABC):
    """Scores items for visitors.

    Attributes:
        name: Model identifier, matching ``cfg.model.name``.
    """

    name: str = "abstract"

    #: Whether the model can score a visitor absent from the train mapping.
    #: Personalised graph models cannot: a cold visitor has no embedding, so
    #: their cold-segment metrics are reported as null rather than invented.
    supports_cold_start: bool = False

    @abstractmethod
    def fit(self, context: ModelContext) -> None:
        """Learn from train data only."""

    @abstractmethod
    def score(self, visitor_indices: np.ndarray) -> np.ndarray:
        """Score every train item for a batch of visitors.

        Args:
            visitor_indices: Visitor matrix indices, or ``-1`` for a cold
                visitor absent from the train mapping.

        Returns:
            Array of shape ``(len(visitor_indices), n_items)``; column ``j`` is
            the score of the item whose matrix index is ``j``. Higher ranks
            higher. Scores are ranking scores, not probabilities.
        """

    def describe(self) -> dict[str, object]:
        """Serialisable description recorded in the run artifact."""
        return {"model": self.name, "supports_cold_start": self.supports_cold_start}


class NotFittedError(RuntimeError):
    """Raised when :meth:`Recommender.score` is called before :meth:`fit`."""
