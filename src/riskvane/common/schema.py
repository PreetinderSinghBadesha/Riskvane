"""Wire format shared between the posture collector and the (future) scoring API."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class OsInfo(BaseModel):
    name: str
    version: str
    kernel: str
    architecture: str


class Package(BaseModel):
    name: str
    version: str


class OpenPort(BaseModel):
    port: int
    protocol: str
    listen_address: str
    pid: int | None = None
    process_name: str | None = None


class SshConfig(BaseModel):
    permit_root_login: str | None = None
    password_authentication: str | None = None
    protocol: str | None = None


class ServiceProbe(BaseModel):
    port: int
    service: str
    unauthenticated: bool
    detail: str | None = None


class FilesystemFinding(BaseModel):
    path: str
    issue: str  # "world_writable" | "suid"


class ProcessSummary(BaseModel):
    total_processes: int
    root_processes: int
    sudo_users: list[str] = Field(default_factory=list)


class PostureFacts(BaseModel):
    vm_id: str
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    os: OsInfo
    packages: list[Package] = Field(default_factory=list)
    network: list[OpenPort] = Field(default_factory=list)
    ssh: SshConfig
    probes: list[ServiceProbe] = Field(default_factory=list)
    filesystem: list[FilesystemFinding] = Field(default_factory=list)
    processes: ProcessSummary


class Severity(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    info = "info"


class Finding(BaseModel):
    id: str
    category: Literal["cve", "config"]
    title: str
    severity: Severity
    description: str
    evidence: str | None = None


class ScanResult(BaseModel):
    score: int
    severity_breakdown: dict[Severity, int]
    findings: list[Finding] = Field(default_factory=list)
    score_explanation: dict[str, float] | None = None
