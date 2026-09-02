from riskvane.common.schema import Finding, OsInfo, PostureFacts, ProcessSummary, SshConfig
from riskvane.posture.scoring import scorer


def _facts() -> PostureFacts:
    return PostureFacts(
        vm_id="test-vm",
        os=OsInfo(name="Ubuntu", version="22.04", kernel="5.15", architecture="x86_64"),
        ssh=SshConfig(),
        processes=ProcessSummary(total_processes=10, root_processes=1),
    )


def _finding(severity: str, suffix: str = "") -> Finding:
    return Finding(
        id=f"finding-{severity}{suffix}",
        category="config",
        title="test finding",
        severity=severity,
        description="test",
    )


def test_sums_points_by_severity(monkeypatch):
    monkeypatch.setattr(scorer.cve_feed, "match", lambda packages, feed: [_finding("critical")])
    monkeypatch.setattr(scorer.rules, "evaluate", lambda facts: [_finding("high"), _finding("low")])

    result = scorer.score(_facts(), feed=[])

    assert result.score == 40 + 20 + 5
    assert result.severity_breakdown["critical"] == 1
    assert result.severity_breakdown["high"] == 1
    assert result.severity_breakdown["low"] == 1
    assert result.severity_breakdown["medium"] == 0
    assert result.severity_breakdown["info"] == 0
    assert len(result.findings) == 3


def test_score_is_capped_at_100(monkeypatch):
    monkeypatch.setattr(scorer.cve_feed, "match", lambda packages, feed: [])
    monkeypatch.setattr(
        scorer.rules, "evaluate", lambda facts: [_finding("critical", str(i)) for i in range(5)]
    )

    result = scorer.score(_facts(), feed=[])

    assert result.score == 100


def test_no_findings_yields_zero_score(monkeypatch):
    monkeypatch.setattr(scorer.cve_feed, "match", lambda packages, feed: [])
    monkeypatch.setattr(scorer.rules, "evaluate", lambda facts: [])

    result = scorer.score(_facts(), feed=[])

    assert result.score == 0
    assert result.findings == []


def test_loads_default_feed_when_none_given(monkeypatch):
    monkeypatch.setattr(scorer.cve_feed, "load_feed", lambda: "sentinel-feed")
    captured = {}

    def fake_match(packages, feed):
        captured["feed"] = feed
        return []

    monkeypatch.setattr(scorer.cve_feed, "match", fake_match)
    monkeypatch.setattr(scorer.rules, "evaluate", lambda facts: [])

    scorer.score(_facts())

    assert captured["feed"] == "sentinel-feed"
