from riskvane.common.schema import (
    FilesystemFinding,
    OsInfo,
    PostureFacts,
    ProcessSummary,
    ServiceProbe,
    SshConfig,
)
from riskvane.posture.scoring import rules


def _facts(**overrides) -> PostureFacts:
    defaults = {
        "vm_id": "test-vm",
        "os": OsInfo(name="Ubuntu", version="22.04", kernel="5.15", architecture="x86_64"),
        "ssh": SshConfig(permit_root_login="no", password_authentication="no", protocol="2"),
        "processes": ProcessSummary(total_processes=10, root_processes=1),
    }
    defaults.update(overrides)
    return PostureFacts(**defaults)


def test_flags_root_login_permitted():
    facts = _facts(ssh=SshConfig(permit_root_login="yes"))

    findings = rules.evaluate(facts)

    assert any(f.id == "ssh-root-login" for f in findings)


def test_does_not_flag_root_login_disabled():
    facts = _facts(ssh=SshConfig(permit_root_login="no"))

    findings = rules.evaluate(facts)

    assert not any(f.id == "ssh-root-login" for f in findings)


def test_flags_password_authentication_enabled():
    facts = _facts(ssh=SshConfig(password_authentication="yes"))

    findings = rules.evaluate(facts)

    assert any(f.id == "ssh-password-auth" for f in findings)


def test_flags_unauthenticated_service_probe():
    facts = _facts(
        probes=[ServiceProbe(port=6379, service="redis", unauthenticated=True, detail="+PONG")]
    )

    findings = rules.evaluate(facts)

    assert len(findings) == 1
    assert findings[0].id == "exposed-unauthenticated-redis-6379"
    assert findings[0].severity == "critical"


def test_does_not_flag_authenticated_service_probe():
    facts = _facts(
        probes=[ServiceProbe(port=6379, service="redis", unauthenticated=False, detail="NOAUTH")]
    )

    findings = rules.evaluate(facts)

    assert not any(f.category == "config" and "exposed-unauthenticated" in f.id for f in findings)


def test_flags_filesystem_findings():
    facts = _facts(
        filesystem=[
            FilesystemFinding(path="/tmp/evil.sh", issue="world_writable"),
            FilesystemFinding(path="/usr/bin/suid_bin", issue="suid"),
        ]
    )

    findings = rules.evaluate(facts)

    ids = {f.id for f in findings}
    assert "filesystem-world_writable-/tmp/evil.sh" in ids
    assert "filesystem-suid-/usr/bin/suid_bin" in ids


def test_clean_facts_produce_no_findings():
    facts = _facts()

    findings = rules.evaluate(facts)

    assert findings == []
