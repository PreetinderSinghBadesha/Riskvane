"""Normalized event wire format shared across all live-attack-detection log sources."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from riskvane.common.schema import Severity


class EventType(str, Enum):
    ssh_auth_success = "ssh_auth_success"
    ssh_auth_failure = "ssh_auth_failure"
    ssh_invalid_user = "ssh_invalid_user"
    ssh_connection_closed = "ssh_connection_closed"
    firewall_connection_logged = "firewall_connection_logged"


class Event(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    vm_id: str
    source: Literal["auth", "firewall"]
    event_type: EventType
    timestamp: datetime
    source_ip: str | None = None
    dest_port: int | None = None
    username: str | None = None
    raw: str


class Alert(BaseModel):
    alert_type: str
    severity: Severity
    vm_id: str
    source_ip: str | None = None
    evidence: str
    raw_event_ids: list[str] = Field(default_factory=list)
    triggered_at: datetime
