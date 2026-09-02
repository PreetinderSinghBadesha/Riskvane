from fastapi.testclient import TestClient

from riskvane.posture.api.app import create_app


def _facts_payload() -> dict:
    return {
        "vm_id": "test-vm",
        "os": {"name": "Ubuntu", "version": "22.04", "kernel": "5.15", "architecture": "x86_64"},
        "packages": [{"name": "bash", "version": "4.2"}],
        "network": [],
        "ssh": {"permit_root_login": "yes", "password_authentication": "no", "protocol": "2"},
        "probes": [],
        "filesystem": [],
        "processes": {"total_processes": 10, "root_processes": 1, "sudo_users": []},
    }


def test_scan_returns_score_and_findings():
    client = TestClient(create_app())

    response = client.post("/scan", json=_facts_payload())

    assert response.status_code == 200
    body = response.json()
    assert "score" in body
    assert "severity_breakdown" in body
    assert isinstance(body["findings"], list)
    assert any(f["id"] == "ssh-root-login" for f in body["findings"])
    assert any(f["id"] == "CVE-2014-6271" for f in body["findings"])


def test_scan_rejects_malformed_payload():
    client = TestClient(create_app())

    response = client.post("/scan", json={"vm_id": "test-vm"})

    assert response.status_code == 422
