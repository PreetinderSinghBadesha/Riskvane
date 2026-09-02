import json

from riskvane.common.schema import (
    OpenPort,
    OsInfo,
    Package,
    ProcessSummary,
    ServiceProbe,
    SshConfig,
)
from riskvane.posture.collector import facts


def test_collect_assembles_and_round_trips_through_json(monkeypatch):
    monkeypatch.setattr(
        facts.os_info, "collect", lambda: OsInfo(name="Ubuntu", version="22.04", kernel="5.15", architecture="x86_64")
    )
    monkeypatch.setattr(facts.packages, "collect", lambda: [Package(name="bash", version="5.1")])
    fake_ports = [OpenPort(port=22, protocol="tcp", listen_address="0.0.0.0", process_name="sshd")]
    monkeypatch.setattr(facts.network, "collect", lambda: fake_ports)
    monkeypatch.setattr(
        facts.ssh_config,
        "collect",
        lambda: SshConfig(permit_root_login="no", password_authentication="no", protocol="2"),
    )
    monkeypatch.setattr(
        facts.probes,
        "collect",
        lambda ports: [ServiceProbe(port=22, service="ssh", unauthenticated=False)],
    )
    monkeypatch.setattr(facts.filesystem, "collect", list)
    monkeypatch.setattr(
        facts.processes, "collect", lambda: ProcessSummary(total_processes=100, root_processes=5)
    )

    result = facts.collect(vm_id="test-vm")

    assert result.vm_id == "test-vm"
    assert result.os.name == "Ubuntu"
    assert result.network[0].port == 22
    assert result.probes[0].port == fake_ports[0].port

    payload = json.loads(result.model_dump_json())
    assert payload["vm_id"] == "test-vm"
    assert payload["packages"][0]["name"] == "bash"
