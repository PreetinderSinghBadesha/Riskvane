"""Generic sliding-window counter: a rolling per-key window of timestamped items."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Hashable
from datetime import datetime, timedelta
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
T = TypeVar("T")


class SlidingWindowCounter(Generic[K, T]):
    def __init__(self, window: timedelta):
        self._window = window
        self._buckets: dict[K, deque[tuple[datetime, T]]] = defaultdict(deque)

    def add(self, key: K, timestamp: datetime, item: T) -> list[T]:
        bucket = self._buckets[key]
        bucket.append((timestamp, item))
        self._prune(bucket, timestamp)
        return [entry for _, entry in bucket]

    def _prune(self, bucket: deque[tuple[datetime, T]], now: datetime) -> None:
        cutoff = now - self._window
        while bucket and bucket[0][0] < cutoff:
            bucket.popleft()
