"""Combines CVE matches and rule-check findings into a single deterministic ScanResult."""

from __future__ import annotations

from riskvane.common.schema import Finding, PostureFacts, ScanResult, Severity
from riskvane.posture.scoring import cve_feed, rules
from riskvane.posture.scoring.cve_feed import CveEntry

_POINTS_BY_SEVERITY = {
    Severity.critical: 40,
    Severity.high: 20,
    Severity.medium: 10,
    Severity.low: 5,
    Severity.info: 1,
}


def total_points(findings: list[Finding]) -> int:
    """Capped risk score for a set of findings, using the same weights as ``score``."""
    return min(100, sum(_POINTS_BY_SEVERITY[finding.severity] for finding in findings))


def score(facts: PostureFacts, feed: list[CveEntry] | None = None) -> ScanResult:
    if feed is None:
        feed = cve_feed.load_feed()

    findings = cve_feed.match(facts.packages, feed) + rules.evaluate(facts)

    breakdown = {severity: 0 for severity in Severity}
    for finding in findings:
        breakdown[finding.severity] += 1

    return ScanResult(
        score=total_points(findings),
        severity_breakdown=breakdown,
        findings=findings,
    )
