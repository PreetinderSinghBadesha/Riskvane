"""Turns a PostureFacts payload into a flat numeric feature vector for future model training."""

from __future__ import annotations

from riskvane.common.schema import PostureFacts


def featurize(facts: PostureFacts) -> dict[str, float]:
    return {
        "num_packages": float(len(facts.packages)),
        "num_open_ports": float(len(facts.network)),
        "permit_root_login": 1.0 if facts.ssh.permit_root_login in ("yes", "without-password") else 0.0,
        "password_authentication_enabled": 1.0 if facts.ssh.password_authentication == "yes" else 0.0,
        "num_unauthenticated_probes": float(sum(1 for p in facts.probes if p.unauthenticated)),
        "num_world_writable_files": float(
            sum(1 for f in facts.filesystem if f.issue == "world_writable")
        ),
        "num_suid_binaries": float(sum(1 for f in facts.filesystem if f.issue == "suid")),
        "total_processes": float(facts.processes.total_processes),
        "root_processes": float(facts.processes.root_processes),
        "num_sudo_users": float(len(facts.processes.sudo_users)),
    }
