from datetime import datetime, timedelta

from riskvane.common.events import Event, EventType
from riskvane.detection.rules import BruteForceRule

_T0 = datetime(2026, 1, 1, 12, 0, 0)  # noqa: DTZ001 -- naive, matches Event.timestamp
_VM_ID = "test-vm"
_IP = "203.0.113.5"


def _event(event_type=EventType.ssh_auth_failure, offset_seconds=0, source_ip=_IP, vm_id=_VM_ID):
    return Event(
        vm_id=vm_id,
        source="auth",
        event_type=event_type,
        timestamp=_T0 + timedelta(seconds=offset_seconds),
        source_ip=source_ip,
        username="root",
        raw="raw log line",
    )


def test_no_alert_below_threshold():
    rule = BruteForceRule(threshold=5)

    alerts = [rule.process(_event(offset_seconds=i)) for i in range(4)]

    assert all(a is None for a in alerts)


def test_fires_exactly_once_when_threshold_crossed():
    rule = BruteForceRule(threshold=5)

    alerts = [rule.process(_event(offset_seconds=i)) for i in range(7)]

    fired = [a for a in alerts if a is not None]
    assert len(fired) == 1
    assert fired[0].alert_type == "ssh_brute_force"
    assert fired[0].vm_id == _VM_ID
    assert fired[0].source_ip == _IP
    assert len(fired[0].raw_event_ids) == 5


def test_refires_after_window_ages_out_and_crosses_again():
    rule = BruteForceRule(window=timedelta(minutes=5), threshold=3)

    first_burst = [rule.process(_event(offset_seconds=i)) for i in range(3)]
    assert sum(1 for a in first_burst if a is not None) == 1

    # far enough later that the first burst has fully aged out of the window
    later = 10 * 60
    second_burst = [rule.process(_event(offset_seconds=later + i)) for i in range(3)]
    assert sum(1 for a in second_burst if a is not None) == 1


def test_ignores_non_triggering_event_types():
    rule = BruteForceRule(threshold=1)

    alert = rule.process(_event(event_type=EventType.ssh_auth_success))

    assert alert is None


def test_ignores_events_without_source_ip():
    rule = BruteForceRule(threshold=1)

    alert = rule.process(_event(source_ip=None))

    assert alert is None


def test_different_ips_tracked_independently():
    rule = BruteForceRule(threshold=2)

    rule.process(_event(offset_seconds=0, source_ip="1.1.1.1"))
    alert = rule.process(_event(offset_seconds=1, source_ip="2.2.2.2"))

    assert alert is None
