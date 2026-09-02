from datetime import datetime, timezone

from riskvane.common.events import EventType
from riskvane.detection.parsers import auth_log

_VM_ID = "test-vm"
_YEAR = 2026


def test_parses_failed_password():
    line = "Jun 14 15:16:01 myhost sshd[1234]: Failed password for root from 203.0.113.5 port 51234 ssh2"

    event = auth_log.parse_line(line, _VM_ID, year=_YEAR)

    assert event.event_type == EventType.ssh_auth_failure
    assert event.username == "root"
    assert event.source_ip == "203.0.113.5"
    assert event.vm_id == _VM_ID
    assert event.source == "auth"
    assert event.timestamp.year == _YEAR
    assert event.timestamp.month == 6
    assert event.timestamp.day == 14


def test_parses_failed_password_for_invalid_user():
    line = (
        "Jun 14 15:16:01 myhost sshd[1234]: Failed password for invalid user admin "
        "from 203.0.113.5 port 51234 ssh2"
    )

    event = auth_log.parse_line(line, _VM_ID, year=_YEAR)

    assert event.event_type == EventType.ssh_auth_failure
    assert event.username == "admin"
    assert event.source_ip == "203.0.113.5"


def test_parses_accepted_publickey():
    line = "Jun 14 15:16:05 myhost sshd[1234]: Accepted publickey for deploy from 198.51.100.7 port 55678 ssh2"

    event = auth_log.parse_line(line, _VM_ID, year=_YEAR)

    assert event.event_type == EventType.ssh_auth_success
    assert event.username == "deploy"
    assert event.source_ip == "198.51.100.7"


def test_parses_invalid_user():
    line = "Jun 14 15:16:10 myhost sshd[1234]: Invalid user test from 203.0.113.5 port 51240"

    event = auth_log.parse_line(line, _VM_ID, year=_YEAR)

    assert event.event_type == EventType.ssh_invalid_user
    assert event.username == "test"
    assert event.source_ip == "203.0.113.5"


def test_parses_connection_closed_by_authenticating_user():
    line = (
        "Jun 14 15:16:15 myhost sshd[1234]: Connection closed by authenticating user admin "
        "203.0.113.5 port 51234 [preauth]"
    )

    event = auth_log.parse_line(line, _VM_ID, year=_YEAR)

    assert event.event_type == EventType.ssh_connection_closed
    assert event.username == "admin"
    assert event.source_ip == "203.0.113.5"


def test_unrecognized_message_returns_none():
    line = "Jun 14 15:16:20 myhost sshd[1234]: Server listening on 0.0.0.0 port 22."

    assert auth_log.parse_line(line, _VM_ID, year=_YEAR) is None


def test_non_sshd_line_returns_none():
    line = "Jun 14 15:16:25 myhost CRON[5678]: (root) CMD (run-parts /etc/cron.hourly)"

    assert auth_log.parse_line(line, _VM_ID, year=_YEAR) is None


def test_parse_lines_filters_and_preserves_order():
    lines = [
        "Jun 14 15:16:20 myhost sshd[1234]: Server listening on 0.0.0.0 port 22.",
        "Jun 14 15:16:01 myhost sshd[1234]: Failed password for root from 203.0.113.5 port 51234 ssh2",
        "Jun 14 15:16:05 myhost sshd[1234]: Accepted publickey for deploy from 198.51.100.7 port 55678 ssh2",
    ]

    events = auth_log.parse_lines(lines, _VM_ID, year=_YEAR)

    assert len(events) == 2
    assert events[0].event_type == EventType.ssh_auth_failure
    assert events[1].event_type == EventType.ssh_auth_success


def test_parse_line_defaults_to_current_year():
    line = "Jun 14 15:16:01 myhost sshd[1234]: Failed password for root from 203.0.113.5 port 51234 ssh2"

    event = auth_log.parse_line(line, _VM_ID)

    assert event.timestamp.year == datetime.now(timezone.utc).year
