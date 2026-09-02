"""Parses iptables/netfilter LOG-target kernel lines into normalized Events.

LOG-target lines look like:
    Jun 14 15:20:01 myhost kernel: [12345.678901] SRC=203.0.113.9 DST=10.0.0.5 LEN=40
    PROTO=TCP SPT=51234 DPT=22 ...

The bracketed kernel uptime and any custom LOG-prefix text (iptables lets admins configure an
arbitrary string, e.g. "IPTABLES-DROP:") vary a lot between setups, so rather than match a fixed
message pattern like auth_log.py does, this extracts every KEY=value token from the message and
requires SRC and DPT to be present. Same naive-timestamp/year-inference caveat as auth_log.py.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime, timezone

from riskvane.common.events import Event, EventType

_ENVELOPE = re.compile(
    r"^(?P<month>\w{3})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+kernel:\s*(?:\[[^\]]*\]\s*)?(?P<message>.*)$"
)

_KV_PATTERN = re.compile(r"(\w+)=(\S+)")


def _parse_timestamp(month: str, day: str, time_str: str, year: int) -> datetime:
    # kernel log lines carry no timezone info, so this is intentionally naive
    return datetime.strptime(  # noqa: DTZ007
        f"{year} {month} {int(day):02d} {time_str}", "%Y %b %d %H:%M:%S"
    )


def parse_line(line: str, vm_id: str, year: int | None = None) -> Event | None:
    envelope_match = _ENVELOPE.match(line.strip())
    if not envelope_match:
        return None

    fields = dict(_KV_PATTERN.findall(envelope_match.group("message")))
    source_ip = fields.get("SRC")
    dest_port = fields.get("DPT")
    if source_ip is None or dest_port is None:
        return None

    try:
        dest_port_int = int(dest_port)
    except ValueError:
        return None

    return Event(
        vm_id=vm_id,
        source="firewall",
        event_type=EventType.firewall_connection_logged,
        timestamp=_parse_timestamp(
            envelope_match.group("month"),
            envelope_match.group("day"),
            envelope_match.group("time"),
            year or datetime.now(timezone.utc).year,
        ),
        source_ip=source_ip,
        dest_port=dest_port_int,
        raw=line,
    )


def parse_lines(lines: Iterable[str], vm_id: str, year: int | None = None) -> list[Event]:
    return [event for line in lines if (event := parse_line(line, vm_id, year=year)) is not None]
