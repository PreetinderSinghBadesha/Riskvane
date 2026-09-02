from riskvane.common.schema import Package
from riskvane.posture.scoring import cve_feed
from riskvane.posture.scoring.cve_feed import CveEntry


def _feed() -> list[CveEntry]:
    return [
        CveEntry(
            package="openssl",
            cve_id="CVE-2022-0778",
            fixed_in="1.1.1n",
            severity="high",
            summary="Infinite loop in BN_mod_sqrt().",
        )
    ]


def test_vulnerable_installed_version_produces_a_finding():
    packages = [Package(name="openssl", version="1.1.1m")]

    findings = cve_feed.match(packages, _feed())

    assert len(findings) == 1
    assert findings[0].id == "CVE-2022-0778"
    assert findings[0].category == "cve"
    assert findings[0].severity == "high"


def test_fixed_version_installed_produces_no_finding():
    packages = [Package(name="openssl", version="1.1.1n")]

    findings = cve_feed.match(packages, _feed())

    assert findings == []


def test_unrelated_package_produces_no_finding():
    packages = [Package(name="curl", version="7.0.0")]

    findings = cve_feed.match(packages, _feed())

    assert findings == []


def test_load_feed_reads_bundled_seed_data():
    feed = cve_feed.load_feed()

    assert len(feed) > 0
    assert any(entry.package == "openssl" for entry in feed)
