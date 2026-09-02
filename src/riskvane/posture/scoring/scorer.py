"""Combines CVE matches and rule-check findings into a single deterministic ScanResult."""

from __future__ import annotations

from riskvane.common.schema import PostureFacts, ScanResult, Severity
from riskvane.posture.scoring import cve_feed, rules
from riskvane.posture.scoring.cve_feed import CveEntry

_POINTS_BY_SEVERITY = {
    Severity.critical: 40,
    Severity.high: 20,
    Severity.medium: 10,
    Severity.low: 5,
    Severity.info: 1,
}


def score(facts: PostureFacts, feed: list[CveEntry] | None = None) -> ScanResult:
    if feed is None:
        feed = cve_feed.load_feed()

    findings = cve_feed.match(facts.packages, feed) + rules.evaluate(facts)

    breakdown = {severity: 0 for severity in Severity}
    for finding in findings:
        breakdown[finding.severity] += 1

    total_points = sum(_POINTS_BY_SEVERITY[finding.severity] for finding in findings)

    return ScanResult(
        score=min(100, total_points),
        severity_breakdown=breakdown,
        findings=findings,
    )
