import pytest

pytest.importorskip("xgboost")
pytest.importorskip("shap")
pytest.importorskip("sklearn")

from riskvane.posture.ml.inference import RiskModel  # noqa: E402


def test_predict_returns_score_in_range_and_full_explanation(trained_model_paths):
    model_path, features_path = trained_model_paths
    model = RiskModel.load(model_path=model_path, features_path=features_path)

    features = {
        "num_packages": 5.0,
        "num_open_ports": 3.0,
        "permit_root_login": 1.0,
        "password_authentication_enabled": 0.0,
        "num_unauthenticated_probes": 0.0,
        "num_world_writable_files": 0.0,
        "num_suid_binaries": 0.0,
        "total_processes": 50.0,
        "root_processes": 2.0,
        "num_sudo_users": 0.0,
    }
    score, explanation = model.predict(features)

    assert isinstance(score, int)
    assert 0 <= score <= 100
    assert set(explanation.keys()) == set(features.keys())
    assert all(isinstance(v, float) for v in explanation.values())
