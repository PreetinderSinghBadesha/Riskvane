"""Shared fixtures for posture/ml tests."""

from __future__ import annotations

import json

import pytest


def _corpus_rows(n: int) -> list[dict]:
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


@pytest.fixture
def synthetic_corpus_path(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _corpus_rows(20)) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def tiny_corpus_path(tmp_path):
    path = tmp_path / "tiny_corpus.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _corpus_rows(3)) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def trained_model_paths(synthetic_corpus_path, tmp_path):
    from riskvane.posture.ml import train

    model_path = tmp_path / "model.json"
    features_path = tmp_path / "model.features.json"
    exit_code = train.main(
        [
            "--data",
            str(synthetic_corpus_path),
            "--model-out",
            str(model_path),
            "--features-out",
            str(features_path),
            "--min-rows",
            "10",
        ]
    )
    assert exit_code == 0
    return model_path, features_path
