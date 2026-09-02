from riskvane.common.schema import (
    FilesystemFinding,
    OpenPort,
    OsInfo,
    Package,
    PostureFacts,
    ProcessSummary,
    ServiceProbe,
    SshConfig,
)
from riskvane.posture.training.features import featurize


def _facts() -> PostureFacts:
    return PostureFacts(
        vm_id="test-vm",
        os=OsInfo(name="Ubuntu", version="22.04", kernel="5.15", architecture="x86_64"),
        packages=[Package(name="bash", version="4.2"), Package(name="curl", version="7.0")],
        network=[OpenPort(port=22, protocol="tcp", listen_address="0.0.0.0")],
        ssh=SshConfig(permit_root_login="yes", password_authentication="yes", protocol="2"),
        probes=[
            ServiceProbe(port=6379, service="redis", unauthenticated=True),
            ServiceProbe(port=27017, service="mongodb", unauthenticated=False),
        ],
        filesystem=[
            FilesystemFinding(path="/tmp/a", issue="world_writable"),
            FilesystemFinding(path="/tmp/b", issue="world_writable"),
            FilesystemFinding(path="/usr/bin/c", issue="suid"),
        ],
        processes=ProcessSummary(total_processes=100, root_processes=5, sudo_users=["alice"]),
    )


def test_featurize_produces_expected_vector():
    features = featurize(_facts())

    assert features == {
        "num_packages": 2.0,
        "num_open_ports": 1.0,
        "permit_root_login": 1.0,
        "password_authentication_enabled": 1.0,
        "num_unauthenticated_probes": 1.0,
        "num_world_writable_files": 2.0,
        "num_suid_binaries": 1.0,
        "total_processes": 100.0,
        "root_processes": 5.0,
        "num_sudo_users": 1.0,
    }


def test_featurize_clean_facts_is_all_zero_except_expected_fields():
    facts = PostureFacts(
        vm_id="clean-vm",
        os=OsInfo(name="Ubuntu", version="22.04", kernel="5.15", architecture="x86_64"),
        ssh=SshConfig(permit_root_login="no", password_authentication="no"),
        processes=ProcessSummary(total_processes=10, root_processes=1),
    )

    features = featurize(facts)

    assert features["permit_root_login"] == 0.0
    assert features["password_authentication_enabled"] == 0.0
    assert features["num_unauthenticated_probes"] == 0.0
    assert features["num_world_writable_files"] == 0.0
    assert features["num_suid_binaries"] == 0.0
    assert features["num_sudo_users"] == 0.0
