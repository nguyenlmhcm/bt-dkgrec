"""Non-personalised popularity baselines.

``popularity``
    The absolute floor: rank items by how often the **target behavior** occurred
    in train. It answers the committee's first question -- does the proposed
    model beat a naive approach at all?

``recent_popularity``
    The same count restricted to a recent window ending at ``T_train``. It
    isolates the contribution of *recency* from the contribution of *graph
    structure*: if recent popularity alone explains the gain, the graph is not
    earning its place.

Both are deterministic -- they ignore the seed entirely -- so their multi-seed
standard deviation is exactly 0. That must be footnoted in the results table so
it is not mistaken for a computation error (docs/DECISIONS.md muc D8).
"""

from __future__ import annotations

import numpy as np

from src.models.base import ModelContext, NotFittedError, Recommender
from src.utils.logging import get_logger

log = get_logger(__name__)


class Popularity(Recommender):
    """Ranks every visitor by global item popularity in train."""

    name = "popularity"
    #: Popularity needs no visitor history, so a cold visitor can be served.
    #: This is also the fallback CLAUDE.md allows for cold users.
    supports_cold_start = True

    def __init__(self) -> None:
        self._scores: np.ndarray | None = None
        self._signal: str = "target"

    def fit(self, context: ModelContext) -> None:
        """Count qualifying train events per item."""
        events = self._select_events(context)
        counts = np.bincount(
            events["item_idx"].to_numpy(), minlength=context.n_items
        ).astype("float32")
        self._scores = counts
        log.info(
            "%s: %s su kien dem duoc, %s item co diem > 0, max = %s",
            self.name, f"{len(events):,}", f"{int((counts > 0).sum()):,}", f"{int(counts.max()):,}",
        )

    def _select_events(self, context: ModelContext):
        """Which train events feed the count."""
        self._signal = context.cfg.model.popularity_signal or "target"
        events = context.train_events
        if self._signal == "target":
            events = events[events["behavior"].isin(context.cfg.data.target_behaviors)]
        return events

    def score(self, visitor_indices: np.ndarray) -> np.ndarray:
        """Return the same popularity vector for every visitor.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
        if self._scores is None:
            raise NotFittedError(f"{self.name} chua duoc fit")
        return np.broadcast_to(self._scores, (len(visitor_indices), len(self._scores)))

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "popularity_signal": self._signal}


class RecentPopularity(Popularity):
    """Popularity restricted to a recent window ending at ``T_train``.

    Inherits everything from :class:`Popularity` and narrows only which events
    are counted, so the two baselines stay directly comparable.
    """

    name = "recent_popularity"

    def __init__(self) -> None:
        super().__init__()
        self._window_days: int | None = None

    def _select_events(self, context: ModelContext):
        events = super()._select_events(context)
        window_days = context.cfg.model.recent_window_days
        if window_days is None:
            raise ValueError("recent_popularity can recent_window_days trong config")
        self._window_days = window_days
        cutoff = context.t_train - window_days * context.cfg.weighting.d_day
        recent = events[events["timestamp"] > cutoff]
        log.info(
            "%s: cua so %d ngay cuoi train giu %s / %s su kien",
            self.name, window_days, f"{len(recent):,}", f"{len(events):,}",
        )
        return recent

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "recent_window_days": self._window_days}


#: Model registry used by ``scripts/03_train.py``.
HEURISTIC_MODELS: dict[str, type[Popularity]] = {
    "popularity": Popularity,
    "recent_popularity": RecentPopularity,
}
