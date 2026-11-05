"""Per-address limit on write requests for the public demo."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable

WINDOW_SECONDS = 60.0


class WriteLimiter:
    """Sliding one-minute window, in process memory."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_minute = per_minute
        self._clock = clock
        self._seen: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, address: str) -> bool:
        now = self._clock()
        seen = self._seen[address]
        while seen and now - seen[0] >= WINDOW_SECONDS:
            seen.popleft()
        if len(seen) >= self.per_minute:
            return False
        seen.append(now)
        return True
