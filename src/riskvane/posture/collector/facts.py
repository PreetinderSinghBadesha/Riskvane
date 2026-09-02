"""Orchestrates all individual fact-gatherers into one PostureFacts payload."""

from __future__ import annotations

from riskvane.common.schema import PostureFacts
from riskvane.posture.collector import (
    filesystem,
    network,
    os_info,
    packages,
    probes,
    processes,
    ssh_config,
)


def collect(vm_id: str) -> PostureFacts:
    open_ports = network.collect()
    return PostureFacts(
        vm_id=vm_id,
        os=os_info.collect(),
        packages=packages.collect(),
        network=open_ports,
        ssh=ssh_config.collect(),
        probes=probes.collect(open_ports),
        filesystem=filesystem.collect(),
        processes=processes.collect(),
    )
