"""Normalized event wire format shared across all live-attack-detection log sources."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel


class EventType(str, Enum):
    ssh_auth_success = "ssh_auth_success"
    ssh_auth_failure = "ssh_auth_failure"
    ssh_invalid_user = "ssh_invalid_user"
    ssh_connection_closed = "ssh_connection_closed"


class Event(BaseModel):
    vm_id: str
    source: Literal["auth"]
    event_type: EventType
    timestamp: datetime
    source_ip: str | None = None
    username: str | None = None
    raw: str
