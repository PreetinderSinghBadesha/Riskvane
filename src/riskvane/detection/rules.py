"""Threshold alert rules evaluated over the sliding-window aggregator."""

from __future__ import annotations

from datetime import timedelta

from riskvane.common.events import Alert, Event, EventType
from riskvane.detection.aggregator import SlidingWindowCounter

_TRIGGERING_EVENT_TYPES = {EventType.ssh_auth_failure, EventType.ssh_invalid_user}


class BruteForceRule:
    alert_type = "ssh_brute_force"
    severity = "high"

    def __init__(self, window: timedelta = timedelta(minutes=5), threshold: int = 5):
        self._window = window
        self._threshold = threshold
        self._counter: SlidingWindowCounter[tuple[str, str], Event] = SlidingWindowCounter(window)
        self._active_keys: set[tuple[str, str]] = set()

    def process(self, event: Event) -> Alert | None:
        if event.event_type not in _TRIGGERING_EVENT_TYPES or event.source_ip is None:
            return None

        key = (event.vm_id, event.source_ip)
        window_events = self._counter.add(key, event.timestamp, event)

        if len(window_events) < self._threshold:
            self._active_keys.discard(key)
            return None

        if key in self._active_keys:
            return None

        self._active_keys.add(key)
        return Alert(
            alert_type=self.alert_type,
            severity=self.severity,
            vm_id=event.vm_id,
            source_ip=event.source_ip,
            evidence=(
                f"{len(window_events)} failed SSH logins from {event.source_ip} "
                f"within {self._window}"
            ),
            raw_event_ids=[e.id for e in window_events],
            triggered_at=event.timestamp,
        )


class PortScanRule:
    alert_type = "port_scan"
    severity = "medium"

    def __init__(self, window: timedelta = timedelta(minutes=1), threshold: int = 10):
        self._window = window
        self._threshold = threshold
        self._counter: SlidingWindowCounter[tuple[str, str], Event] = SlidingWindowCounter(window)
        self._active_keys: set[tuple[str, str]] = set()

    def process(self, event: Event) -> Alert | None:
        if (
            event.event_type != EventType.firewall_connection_logged
            or event.source_ip is None
            or event.dest_port is None
        ):
            return None

        key = (event.vm_id, event.source_ip)
        window_events = self._counter.add(key, event.timestamp, event)
        distinct_ports = {e.dest_port for e in window_events}

        if len(distinct_ports) < self._threshold:
            self._active_keys.discard(key)
            return None

        if key in self._active_keys:
            return None

        self._active_keys.add(key)
        return Alert(
            alert_type=self.alert_type,
            severity=self.severity,
            vm_id=event.vm_id,
            source_ip=event.source_ip,
            evidence=(
                f"{len(distinct_ports)} distinct ports hit by {event.source_ip} "
                f"within {self._window}"
            ),
            raw_event_ids=[e.id for e in window_events],
            triggered_at=event.timestamp,
        )
