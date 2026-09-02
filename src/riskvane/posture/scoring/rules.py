"""Static hardening rule checks, operating only on facts the collector already gathers."""

from __future__ import annotations

from riskvane.common.schema import Finding, PostureFacts


def _check_root_login(facts: PostureFacts) -> Finding | None:
    if facts.ssh.permit_root_login in ("yes", "without-password"):
        return Finding(
            id="ssh-root-login",
            category="config",
            title="SSH root login is permitted",
            severity="high",
            description="sshd_config PermitRootLogin allows logging in directly as root over SSH.",
            evidence=f"PermitRootLogin {facts.ssh.permit_root_login}",
        )
    return None


def _check_password_authentication(facts: PostureFacts) -> Finding | None:
    if facts.ssh.password_authentication == "yes":
        return Finding(
            id="ssh-password-auth",
            category="config",
            title="SSH password authentication is enabled",
            severity="medium",
            description="sshd_config PasswordAuthentication allows password-based login instead of key-only auth.",
            evidence="PasswordAuthentication yes",
        )
    return None


def _check_unauthenticated_services(facts: PostureFacts) -> list[Finding]:
    return [
        Finding(
            id=f"exposed-unauthenticated-{probe.service}-{probe.port}",
            category="config",
            title=f"{probe.service} on port {probe.port} requires no authentication",
            severity="critical",
            description="A network-reachable service accepted commands without any authentication.",
            evidence=probe.detail,
        )
        for probe in facts.probes
        if probe.unauthenticated
    ]


def _check_filesystem_findings(facts: PostureFacts) -> list[Finding]:
    severity_by_issue = {"world_writable": "medium", "suid": "low"}
    return [
        Finding(
            id=f"filesystem-{finding.issue}-{finding.path}",
            category="config",
            title=f"{finding.issue.replace('_', ' ')} file: {finding.path}",
            severity=severity_by_issue.get(finding.issue, "info"),
            description=f"Filesystem scan flagged {finding.path} as {finding.issue}.",
            evidence=finding.path,
        )
        for finding in facts.filesystem
    ]


def evaluate(facts: PostureFacts) -> list[Finding]:
    findings: list[Finding] = []
    for single in (_check_root_login(facts), _check_password_authentication(facts)):
        if single is not None:
            findings.append(single)
    findings.extend(_check_unauthenticated_services(facts))
    findings.extend(_check_filesystem_findings(facts))
    return findings
