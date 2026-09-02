"""Loads a trained risk model and predicts a score + per-feature SHAP explanation.

Requires the optional `ml` extra (xgboost) — this module is only ever imported lazily by callers
that actually want to use the model, so the base install never needs the dependency.

SHAP values come from XGBoost's own TreeSHAP implementation (`pred_contribs=True`) rather than the
separate `shap` package: it's the same algorithm (shap.TreeExplainer delegates to it for XGBoost
models) without pulling in numba/llvmlite.
"""

from __future__ import annotations

import json
from pathlib import Path

import xgboost as xgb

DEFAULT_MODEL_PATH = Path("var/models/risk_model.json")
DEFAULT_FEATURES_PATH = Path("var/models/risk_model.features.json")


class RiskModel:
    def __init__(self, booster: xgb.Booster, feature_names: list[str]):
        self._booster = booster
        self._feature_names = feature_names

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

        # pred_contribs returns one column per feature plus a trailing bias/base-value column
        contributions = self._booster.predict(dmatrix, pred_contribs=True)[0]
        explanation = dict(
            zip(self._feature_names, (float(v) for v in contributions[:-1]), strict=True)
        )

        return score, explanation
