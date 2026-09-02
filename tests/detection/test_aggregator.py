from datetime import datetime, timedelta

from riskvane.detection.aggregator import SlidingWindowCounter

_T0 = datetime(2026, 1, 1, 12, 0, 0)  # noqa: DTZ001 -- naive, matches Event.timestamp


def test_items_within_window_accumulate():
    counter = SlidingWindowCounter(window=timedelta(minutes=5))

    counter.add("k", _T0, "a")
    counter.add("k", _T0 + timedelta(minutes=1), "b")
    result = counter.add("k", _T0 + timedelta(minutes=2), "c")

    assert result == ["a", "b", "c"]


def test_items_outside_window_are_pruned():
    counter = SlidingWindowCounter(window=timedelta(minutes=5))

    counter.add("k", _T0, "a")
    counter.add("k", _T0 + timedelta(minutes=1), "b")
    result = counter.add("k", _T0 + timedelta(minutes=10), "c")

    assert result == ["c"]


def test_keys_are_independent():
    counter = SlidingWindowCounter(window=timedelta(minutes=5))

    counter.add("k1", _T0, "a")
    result_k2 = counter.add("k2", _T0, "b")

    assert result_k2 == ["b"]


def test_boundary_item_exactly_at_cutoff_is_kept():
    counter = SlidingWindowCounter(window=timedelta(minutes=5))

    counter.add("k", _T0, "a")
    result = counter.add("k", _T0 + timedelta(minutes=5), "b")

    assert result == ["a", "b"]


def test_item_just_past_cutoff_is_pruned():
    counter = SlidingWindowCounter(window=timedelta(minutes=5))

    counter.add("k", _T0, "a")
    result = counter.add("k", _T0 + timedelta(minutes=5, seconds=1), "b")

    assert result == ["b"]
