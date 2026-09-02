from riskvane.common.events import EventType
from riskvane.detection.parsers import firewall_log

_VM_ID = "test-vm"
_YEAR = 2026


def test_parses_a_log_target_line_with_bracketed_uptime():
    line = (
        "Jun 14 15:20:01 myhost kernel: [12345.678901] IN=eth0 OUT= "
        "SRC=203.0.113.9 DST=10.0.0.5 LEN=40 PROTO=TCP SPT=51234 DPT=22 WINDOW=1024"
    )

    event = firewall_log.parse_line(line, _VM_ID, year=_YEAR)

    assert event.event_type == EventType.firewall_connection_logged
    assert event.source == "firewall"
    assert event.source_ip == "203.0.113.9"
    assert event.dest_port == 22
    assert event.vm_id == _VM_ID
    assert event.timestamp.year == _YEAR
    assert event.timestamp.month == 6
    assert event.timestamp.day == 14


def test_parses_a_log_target_line_with_custom_prefix_and_no_bracket():
    line = (
        "Jun 14 15:20:05 myhost kernel: IPTABLES-DROP: IN=eth0 "
        "SRC=198.51.100.2 DST=10.0.0.5 PROTO=TCP DPT=6379"
    )

    event = firewall_log.parse_line(line, _VM_ID, year=_YEAR)

    assert event.source_ip == "198.51.100.2"
    assert event.dest_port == 6379


def test_kernel_line_without_src_or_dpt_returns_none():
    line = "Jun 14 15:20:10 myhost kernel: microcode: updated early to revision 0x1, date = 2023-01-01"

    assert firewall_log.parse_line(line, _VM_ID, year=_YEAR) is None


def test_non_kernel_line_returns_none():
    line = "Jun 14 15:20:15 myhost sshd[1234]: Accepted publickey for deploy from 198.51.100.7 port 55678 ssh2"

    assert firewall_log.parse_line(line, _VM_ID, year=_YEAR) is None


def test_parse_lines_filters_and_preserves_order():
    lines = [
        "Jun 14 15:20:10 myhost kernel: microcode: updated early to revision 0x1, date = 2023-01-01",
        "Jun 14 15:20:01 myhost kernel: [1.0] SRC=203.0.113.9 DST=10.0.0.5 PROTO=TCP DPT=22",
        "Jun 14 15:20:02 myhost kernel: [2.0] SRC=203.0.113.9 DST=10.0.0.5 PROTO=TCP DPT=23",
    ]

    events = firewall_log.parse_lines(lines, _VM_ID, year=_YEAR)

    assert len(events) == 2
    assert events[0].dest_port == 22
    assert events[1].dest_port == 23
