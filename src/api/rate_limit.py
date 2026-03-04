from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from time import monotonic


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    retry_after_s: float


class SlidingWindowRateLimiter:
    """Process-local, thread-safe sliding-window request limiter."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, *, key: str, limit: int, window_s: int) -> RateLimitResult:
        if limit <= 0 or window_s <= 0:
            return RateLimitResult(allowed=True, retry_after_s=0.0)

        now = monotonic()
        cutoff = now - window_s

        with self._lock:
            timestamps = self._hits[key]
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= limit:
                oldest = timestamps[0]
                retry_after_s = max(window_s - (now - oldest), 0.001)
                return RateLimitResult(allowed=False, retry_after_s=retry_after_s)

            timestamps.append(now)
            return RateLimitResult(allowed=True, retry_after_s=0.0)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
