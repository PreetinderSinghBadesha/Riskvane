from datetime import datetime

from riskvane.common.events import Alert, Event, EventType
from riskvane.detection.engine import DetectionEngine

_T0 = datetime(2026, 1, 1, 12, 0, 0)  # noqa: DTZ001 -- naive, matches Event.timestamp


def _event(vm_id="test-vm"):
    return Event(
        vm_id=vm_id,
        source="auth",
        event_type=EventType.ssh_auth_failure,
        timestamp=_T0,
        source_ip="203.0.113.5",
        raw="raw log line",
    )


class _AlwaysFiresRule:
    def process(self, event: Event) -> Alert | None:
        return Alert(
            alert_type="stub_alert",
            severity="low",
            vm_id=event.vm_id,
            source_ip=event.source_ip,
            evidence="stub",
            raw_event_ids=[event.id],
            triggered_at=event.timestamp,
        )


class _NeverFiresRule:
    def process(self, event: Event) -> Alert | None:
        return None


def test_process_event_returns_and_stores_fired_alerts():
    engine = DetectionEngine(rules=[_AlwaysFiresRule()])

    fired = engine.process_event(_event(vm_id="vm-a"))

    assert len(fired) == 1
    assert fired[0].alert_type == "stub_alert"
    assert engine.get_alerts("vm-a") == fired


def test_process_event_with_no_firing_rules_returns_empty():
    engine = DetectionEngine(rules=[_NeverFiresRule()])

    fired = engine.process_event(_event(vm_id="vm-a"))

    assert fired == []
    assert engine.get_alerts("vm-a") == []


def test_alerts_are_scoped_per_vm_id():
    engine = DetectionEngine(rules=[_AlwaysFiresRule()])

    engine.process_event(_event(vm_id="vm-a"))
    engine.process_event(_event(vm_id="vm-b"))

    assert len(engine.get_alerts("vm-a")) == 1
    assert len(engine.get_alerts("vm-b")) == 1
    assert engine.get_alerts("vm-c") == []


def test_multiple_rules_can_each_fire():
    engine = DetectionEngine(rules=[_AlwaysFiresRule(), _AlwaysFiresRule()])

    fired = engine.process_event(_event(vm_id="vm-a"))

    assert len(fired) == 2
    assert len(engine.get_alerts("vm-a")) == 2
