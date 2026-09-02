from riskvane.posture.scoring import cve_feed, scorer, version
from riskvane.posture.training.synthetic import (
    _VULNERABLE_VERSIONS,
    generate_facts,
    generate_profiles,
)


def test_generate_profiles_returns_64_unique_profiles():
    profiles = generate_profiles()

    assert len(profiles) == 64
    unique = {tuple(sorted(p.items())) for p in profiles}
    assert len(unique) == 64
    assert {k: False for k in profiles[0]} in profiles
    assert {k: True for k in profiles[0]} in profiles


def test_generate_facts_returns_64_facts_with_unique_vm_ids():
    facts = generate_facts()

    assert len(facts) == 64
    assert len({f.vm_id for f in facts}) == 64


def test_every_current_feed_entry_has_a_vulnerable_version_mapping():
    feed = cve_feed.load_feed()

    for entry in feed:
        assert entry.package in _VULNERABLE_VERSIONS, (
            f"synthetic.py's _VULNERABLE_VERSIONS is missing an entry for {entry.package!r} "
            "-- update it to match the current CVE feed"
        )
        assert version.is_older_than(_VULNERABLE_VERSIONS[entry.package], entry.fixed_in)


def test_all_clean_profile_scores_zero():
    clean_facts = next(
        f
        for f, p in zip(generate_facts(), generate_profiles(), strict=True)
        if not any(p.values())
    )
    feed = cve_feed.load_feed()

    result = scorer.score(clean_facts, feed)

    assert result.score == 0
    assert result.findings == []


def test_all_triggered_profile_scores_100_with_every_finding_category():
    vulnerable_facts = next(
        f
        for f, p in zip(generate_facts(), generate_profiles(), strict=True)
        if all(p.values())
    )
    feed = cve_feed.load_feed()

    result = scorer.score(vulnerable_facts, feed)

    assert result.score == 100
    finding_ids = {f.id for f in result.findings}
    assert "ssh-root-login" in finding_ids
    assert "ssh-password-auth" in finding_ids
    assert any(fid.startswith("exposed-unauthenticated-") for fid in finding_ids)
    assert any(fid.startswith("filesystem-world_writable-") for fid in finding_ids)
    assert any(fid.startswith("filesystem-suid-") for fid in finding_ids)
    assert all(entry.cve_id in finding_ids for entry in feed)


def test_full_corpus_spans_the_severity_range():
    feed = cve_feed.load_feed()

    scores = [scorer.score(facts, feed).score for facts in generate_facts()]

    assert min(scores) == 0
    assert max(scores) == 100
