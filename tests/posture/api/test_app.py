import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from riskvane.posture.api.app import create_router


def _synthetic_corpus_rows(n: int = 20) -> list[dict]:
    rows = []
    for i in range(n):
        num_open_ports = i % 5
        permit_root_login = float(i % 2)
        label = min(100, int(num_open_ports * 5 + permit_root_login * 30))
        rows.append(
            {
                "vm_id": f"vm-{i}",
                "recorded_at": "2026-01-01T00:00:00+00:00",
                "source": "synthetic",
                "features": {
                    "num_packages": float(i),
                    "num_open_ports": float(num_open_ports),
                    "permit_root_login": permit_root_login,
                    "password_authentication_enabled": 0.0,
                    "num_unauthenticated_probes": 0.0,
                    "num_world_writable_files": 0.0,
                    "num_suid_binaries": 0.0,
                    "total_processes": 50.0,
                    "root_processes": 2.0,
                    "num_sudo_users": 0.0,
                },
                "label": label,
                "severity_breakdown": {
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                    "info": 0,
                },
            }
        )
    return rows


def _test_app(training_log_path) -> FastAPI:
    app = FastAPI()
    app.include_router(
        create_router(
            training_log_path=training_log_path,
            model_path=training_log_path.parent / "missing-model.json",
        )
    )
    return app


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


def test_scan_returns_score_and_findings(tmp_path):
    client = TestClient(_test_app(tmp_path / "training.jsonl"))

    response = client.post("/scan", json=_facts_payload())

    assert response.status_code == 200
    body = response.json()
    assert "score" in body
    assert "severity_breakdown" in body
    assert isinstance(body["findings"], list)
    assert any(f["id"] == "ssh-root-login" for f in body["findings"])
    assert any(f["id"] == "CVE-2014-6271" for f in body["findings"])


def test_scan_rejects_malformed_payload(tmp_path):
    client = TestClient(_test_app(tmp_path / "training.jsonl"))

    response = client.post("/scan", json={"vm_id": "test-vm"})

    assert response.status_code == 422


def test_scan_records_a_training_data_row(tmp_path):
    training_log_path = tmp_path / "training.jsonl"
    client = TestClient(_test_app(training_log_path))

    client.post("/scan", json=_facts_payload())

    lines = training_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["vm_id"] == "test-vm"
    assert isinstance(row["label"], int)


def test_scan_use_model_falls_back_to_rule_based_when_no_model_configured(tmp_path):
    client = TestClient(_test_app(tmp_path / "training.jsonl"))

    response = client.post("/scan", params={"use_model": "true"}, json=_facts_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["score_explanation"] is None
    assert any(f["id"] == "ssh-root-login" for f in body["findings"])


def test_scan_use_model_returns_model_score_and_explanation_when_model_exists(tmp_path):
    pytest.importorskip("xgboost")
    from riskvane.posture.ml import train

    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "\n".join(json.dumps(r) for r in _synthetic_corpus_rows()) + "\n", encoding="utf-8"
    )
    model_path = tmp_path / "model.json"
    features_path = tmp_path / "model.features.json"
    exit_code = train.main(
        [
            "--data",
            str(corpus_path),
            "--model-out",
            str(model_path),
            "--features-out",
            str(features_path),
            "--min-rows",
            "10",
        ]
    )
    assert exit_code == 0

    app = FastAPI()
    app.include_router(
        create_router(
            training_log_path=tmp_path / "training.jsonl",
            model_path=model_path,
            features_path=features_path,
        )
    )
    client = TestClient(app)

    plain_response = client.post("/scan", json=_facts_payload())
    model_response = client.post("/scan", params={"use_model": "true"}, json=_facts_payload())

    assert model_response.status_code == 200
    body = model_response.json()
    assert body["score_explanation"]
    assert body["findings"] == plain_response.json()["findings"]
