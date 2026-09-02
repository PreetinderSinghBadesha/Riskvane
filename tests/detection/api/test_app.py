from fastapi import FastAPI
from fastapi.testclient import TestClient

from riskvane.detection.api.app import create_router


def _test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(create_router())
    return app


def _failure_event(offset_seconds: int) -> dict:
    minute = offset_seconds // 60
    second = offset_seconds % 60
    return {
        "vm_id": "test-vm",
        "source": "auth",
        "event_type": "ssh_auth_failure",
        "timestamp": f"2026-01-01T12:{minute:02d}:{second:02d}",
        "source_ip": "203.0.113.5",
        "username": "root",
        "raw": "raw log line",
    }


def test_posting_a_burst_of_failures_fires_a_brute_force_alert():
    client = TestClient(_test_app())
    events = [_failure_event(i) for i in range(6)]

    response = client.post("/events", json=events)

    assert response.status_code == 200
    fired = response.json()
    assert len(fired) == 1
    assert fired[0]["alert_type"] == "ssh_brute_force"
    assert fired[0]["vm_id"] == "test-vm"


def test_alert_is_retrievable_via_get_alerts():
    client = TestClient(_test_app())
    events = [_failure_event(i) for i in range(6)]
    client.post("/events", json=events)

    response = client.get("/alerts", params={"vm_id": "test-vm"})

    assert response.status_code == 200
    alerts = response.json()
    assert len(alerts) == 1
    assert alerts[0]["alert_type"] == "ssh_brute_force"


def test_unrelated_vm_id_has_no_alerts():
    client = TestClient(_test_app())
    events = [_failure_event(i) for i in range(6)]
    client.post("/events", json=events)

    response = client.get("/alerts", params={"vm_id": "some-other-vm"})

    assert response.status_code == 200
    assert response.json() == []


def test_small_batch_below_threshold_fires_nothing():
    client = TestClient(_test_app())
    events = [_failure_event(i) for i in range(2)]

    response = client.post("/events", json=events)

    assert response.status_code == 200
    assert response.json() == []
