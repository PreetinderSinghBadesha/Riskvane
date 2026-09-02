import json

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

from riskvane.posture.ml import train  # noqa: E402


def test_train_writes_model_and_features_artifacts(synthetic_corpus_path, tmp_path):
    model_out = tmp_path / "model.json"
    features_out = tmp_path / "model.features.json"

    exit_code = train.main(
        [
            "--data",
            str(synthetic_corpus_path),
            "--model-out",
            str(model_out),
            "--features-out",
            str(features_out),
            "--min-rows",
            "10",
        ]
    )

    assert exit_code == 0
    assert model_out.exists()
    assert features_out.exists()
    feature_names = json.loads(features_out.read_text(encoding="utf-8"))
    assert "num_open_ports" in feature_names
    assert "permit_root_login" in feature_names


def test_train_rejects_a_too_small_corpus(tiny_corpus_path, tmp_path, capsys):
    model_out = tmp_path / "model.json"
    features_out = tmp_path / "model.features.json"

    exit_code = train.main(
        [
            "--data",
            str(tiny_corpus_path),
            "--model-out",
            str(model_out),
            "--features-out",
            str(features_out),
            "--min-rows",
            "10",
        ]
    )

    assert exit_code == 1
    assert not model_out.exists()
    assert "at least" in capsys.readouterr().err
