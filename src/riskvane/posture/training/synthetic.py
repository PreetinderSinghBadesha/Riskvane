"""Generates a synthetic PostureFacts corpus spanning severity extremes for step 8's training data.

Enumerates every combination (2^6 = 64) of six independent risk dimensions the existing
rule/CVE scorer actually reacts to, rather than hand-picking scenarios or random sampling — a
small, fully deterministic design that hits every partial combination from all-clean (score 0)
to all-triggered (score capped at 100).
"""

from __future__ import annotations

import itertools

from riskvane.common.schema import (
    FilesystemFinding,
    OsInfo,
    Package,
    PostureFacts,
    ProcessSummary,
    ServiceProbe,
    SshConfig,
)
from riskvane.posture.scoring import cve_feed

DIMENSIONS = (
    "root_login",
    "password_auth",
    "cve_exposure",
    "unauthenticated_service",
    "world_writable_files",
    "suid_binaries",
)

# A version older than each current CVE-feed entry's fixed_in. Keep in sync with
# src/riskvane/data/cve_feed.json — test_synthetic.py fails loudly if a package is missing here.
_VULNERABLE_VERSIONS = {
    "openssl": "1.1.1m",
    "bash": "4.2",
    "openssh-server": "7.7",
    "sudo": "1.9.4",
    "curl": "8.3.0",
}


def generate_profiles() -> list[dict[str, bool]]:
    return [
        dict(zip(DIMENSIONS, combo, strict=True))
        for combo in itertools.product([False, True], repeat=len(DIMENSIONS))
    ]


def build_facts(vm_id: str, profile: dict[str, bool]) -> PostureFacts:
    packages = []
    if profile["cve_exposure"]:
        packages = [
            Package(name=entry.package, version=_VULNERABLE_VERSIONS[entry.package])
            for entry in cve_feed.load_feed()
        ]

    probes = []
    if profile["unauthenticated_service"]:
        probes = [
            ServiceProbe(port=6379, service="redis", unauthenticated=True, detail="+PONG"),
            ServiceProbe(
                port=27017,
                service="mongodb",
                unauthenticated=True,
                detail="tcp connect succeeded with no auth challenge",
            ),
        ]

    filesystem = []
    if profile["world_writable_files"]:
        filesystem += [
            FilesystemFinding(path=f"/tmp/writable{i}", issue="world_writable") for i in range(5)
        ]
    if profile["suid_binaries"]:
        filesystem += [
            FilesystemFinding(path=f"/usr/bin/suid{i}", issue="suid") for i in range(5)
        ]

    return PostureFacts(
        vm_id=vm_id,
        os=OsInfo(name="Ubuntu", version="22.04", kernel="5.15", architecture="x86_64"),
        packages=packages,
        network=[],
        ssh=SshConfig(
            permit_root_login="yes" if profile["root_login"] else "no",
            password_authentication="yes" if profile["password_auth"] else "no",
            protocol="2",
        ),
        probes=probes,
        filesystem=filesystem,
        processes=ProcessSummary(total_processes=100, root_processes=2, sudo_users=[]),
    )


def generate_facts() -> list[PostureFacts]:
    return [
        build_facts(f"synthetic-{i:03d}", profile)
        for i, profile in enumerate(generate_profiles())
    ]
