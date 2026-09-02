"""Parses sshd auth-log lines into normalized Event objects.

Syslog-style auth.log lines have no year in the timestamp (e.g. "Jun 14 15:16:01 host sshd[123]:
..."), so callers may pass one explicitly; otherwise it defaults to the current year. This means
logs spanning a Dec/Jan rollover can be mis-dated — an accepted limitation for v0.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime, timezone

from riskvane.common.events import Event, EventType

_ENVELOPE = re.compile(
    r"^(?P<month>\w{3})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+sshd\[\d+\]:\s+(?P<message>.*)$"
)

_MESSAGE_PATTERNS: list[tuple[EventType, re.Pattern]] = [
    (
        EventType.ssh_auth_failure,
        re.compile(r"^Failed password for (?:invalid user )?(?P<user>\S+) from (?P<ip>\S+) port \d+"),
    ),
    (
        EventType.ssh_auth_success,
        re.compile(r"^Accepted \S+ for (?P<user>\S+) from (?P<ip>\S+) port \d+"),
    ),
    (
        EventType.ssh_invalid_user,
        re.compile(r"^Invalid user (?P<user>\S+) from (?P<ip>\S+)"),
    ),
    (
        EventType.ssh_connection_closed,
        re.compile(
            r"^Connection closed by (?:authenticating user (?P<user>\S+) )?(?P<ip>\S+) port \d+"
        ),
    ),
]


def _parse_timestamp(month: str, day: str, time_str: str, year: int) -> datetime:
    # syslog lines carry no timezone info, so this is intentionally naive
    return datetime.strptime(  # noqa: DTZ007
        f"{year} {month} {int(day):02d} {time_str}", "%Y %b %d %H:%M:%S"
    )


def parse_line(line: str, vm_id: str, year: int | None = None) -> Event | None:
    envelope_match = _ENVELOPE.match(line.strip())
    if not envelope_match:
        return None

    message = envelope_match.group("message")
    for event_type, pattern in _MESSAGE_PATTERNS:
        message_match = pattern.match(message)
        if not message_match:
            continue
        groups = message_match.groupdict()
        return Event(
            vm_id=vm_id,
            source="auth",
            event_type=event_type,
            timestamp=_parse_timestamp(
                envelope_match.group("month"),
                envelope_match.group("day"),
                envelope_match.group("time"),
                year or datetime.now(timezone.utc).year,
            ),
            source_ip=groups.get("ip"),
            username=groups.get("user"),
            raw=line,
        )
    return None


def parse_lines(lines: Iterable[str], vm_id: str, year: int | None = None) -> list[Event]:
    return [event for line in lines if (event := parse_line(line, vm_id, year=year)) is not None]
