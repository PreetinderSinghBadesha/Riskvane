"""Loads a trained risk model and predicts a score + per-feature SHAP explanation.

Requires the optional `ml` extra (xgboost, shap) — this module is only ever imported lazily by
callers that actually want to use the model, so the base install never needs these dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

import shap
import xgboost as xgb

DEFAULT_MODEL_PATH = Path("var/models/risk_model.json")
DEFAULT_FEATURES_PATH = Path("var/models/risk_model.features.json")


class RiskModel:
    def __init__(self, booster: xgb.Booster, feature_names: list[str]):
        self._booster = booster
        self._feature_names = feature_names
        self._explainer = shap.TreeExplainer(booster)

    @classmethod
    def load(
        cls,
        model_path: Path = DEFAULT_MODEL_PATH,
        features_path: Path = DEFAULT_FEATURES_PATH,
    ) -> RiskModel:
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        feature_names = json.loads(Path(features_path).read_text(encoding="utf-8"))
        return cls(booster, feature_names)

    def predict(self, features: dict[str, float]) -> tuple[int, dict[str, float]]:
        row = [[features[name] for name in self._feature_names]]
        dmatrix = xgb.DMatrix(row, feature_names=self._feature_names)

        raw_score = float(self._booster.predict(dmatrix)[0])
        score = max(0, min(100, round(raw_score)))

        shap_values = self._explainer.shap_values(dmatrix)[0]
        explanation = dict(zip(self._feature_names, (float(v) for v in shap_values), strict=True))

        return score, explanation
