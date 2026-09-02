"""Wires normalized events through alert rules and stores/serves alerts per VM."""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from riskvane.common.events import Alert, Event
from riskvane.detection.rules import BruteForceRule, PortScanRule


class Rule(Protocol):
    def process(self, event: Event) -> Alert | None: ...


class DetectionEngine:
    def __init__(self, rules: list[Rule] | None = None):
        self._rules: list[Rule] = (
            rules if rules is not None else [BruteForceRule(), PortScanRule()]
        )
        self._alerts: dict[str, list[Alert]] = defaultdict(list)

    def process_event(self, event: Event) -> list[Alert]:
        fired: list[Alert] = []
        for rule in self._rules:
            alert = rule.process(event)
            if alert is not None:
                fired.append(alert)
                self._alerts[event.vm_id].append(alert)
        return fired

    def get_alerts(self, vm_id: str) -> list[Alert]:
        return list(self._alerts.get(vm_id, []))
