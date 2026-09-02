"""Loads the locally cached CVE feed and matches it against installed packages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

from riskvane.common.schema import Finding, Package, Severity
from riskvane.posture.scoring import version

_DEFAULT_FEED_PATH = resources.files("riskvane.data").joinpath("cve_feed.json")


@dataclass(frozen=True)
class CveEntry:
    package: str
    cve_id: str
    fixed_in: str
    severity: Severity
    summary: str


def load_feed(path=_DEFAULT_FEED_PATH) -> list[CveEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        CveEntry(
            package=entry["package"],
            cve_id=entry["cve_id"],
            fixed_in=entry["fixed_in"],
            severity=Severity(entry["severity"]),
            summary=entry["summary"],
        )
        for entry in raw
    ]


def match(packages: list[Package], feed: list[CveEntry]) -> list[Finding]:
    findings: list[Finding] = []
    by_package: dict[str, list[CveEntry]] = {}
    for entry in feed:
        by_package.setdefault(entry.package, []).append(entry)

    for package in packages:
        for entry in by_package.get(package.name, []):
            if version.is_older_than(package.version, entry.fixed_in):
                findings.append(
                    Finding(
                        id=entry.cve_id,
                        category="cve",
                        title=f"{entry.cve_id} in {package.name} {package.version}",
                        severity=entry.severity,
                        description=entry.summary,
                        evidence=f"installed version {package.version}, fixed in {entry.fixed_in}",
                    )
                )
    return findings
