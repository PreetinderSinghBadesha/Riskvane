from fastapi.testclient import TestClient

from riskvane.api.app import create_app


def _facts_payload() -> dict:
    return {
        "vm_id": "unified-vm",
        "os": {"name": "Ubuntu", "version": "22.04", "kernel": "5.15", "architecture": "x86_64"},
        "packages": [],
        "network": [],
        "ssh": {"permit_root_login": "no", "password_authentication": "no", "protocol": "2"},
        "probes": [],
        "filesystem": [],
        "processes": {"total_processes": 10, "root_processes": 1, "sudo_users": []},
    }


def _failure_event(offset_seconds: int) -> dict:
    return {
        "vm_id": "unified-vm",
        "source": "auth",
        "event_type": "ssh_auth_failure",
        "timestamp": f"2026-01-01T12:00:{offset_seconds:02d}",
        "source_ip": "203.0.113.5",
        "username": "root",
        "raw": "raw log line",
    }


def test_scan_is_reachable_on_the_unified_app():
    client = TestClient(create_app())

    response = client.post("/scan", json=_facts_payload())

    assert response.status_code == 200
    body = response.json()
    assert "score" in body
    assert "severity_breakdown" in body


def test_events_and_alerts_are_reachable_on_the_same_unified_app():
    client = TestClient(create_app())
    events = [_failure_event(i) for i in range(6)]

    post_response = client.post("/events", json=events)
    get_response = client.get("/alerts", params={"vm_id": "unified-vm"})

    assert post_response.status_code == 200
    assert len(post_response.json()) == 1
    assert get_response.status_code == 200
    assert len(get_response.json()) == 1
    assert get_response.json()[0]["alert_type"] == "ssh_brute_force"
