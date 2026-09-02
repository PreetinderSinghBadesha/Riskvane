"""v0 posture API: POST /scan."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from riskvane.common.schema import PostureFacts, ScanResult
from riskvane.posture.scoring import cve_feed, scorer
from riskvane.posture.training.features import featurize
from riskvane.posture.training.logger import TrainingDataLogger


def _try_load_model(model_path: Path | None, features_path: Path | None):
    try:
        from riskvane.posture.ml.inference import DEFAULT_MODEL_PATH, RiskModel
    except ImportError:
        return None

    resolved_model_path = model_path or DEFAULT_MODEL_PATH
    if not resolved_model_path.exists():
        return None

    kwargs = {}
    if model_path is not None:
        kwargs["model_path"] = model_path
    if features_path is not None:
        kwargs["features_path"] = features_path
    return RiskModel.load(**kwargs)


def create_router(
    training_log_path: Path | None = None,
    model_path: Path | None = None,
    features_path: Path | None = None,
) -> APIRouter:
    router = APIRouter()
    feed = cve_feed.load_feed()
    training_logger = (
        TrainingDataLogger(training_log_path) if training_log_path else TrainingDataLogger()
    )
    risk_model = _try_load_model(model_path, features_path)

    @router.post("/scan", response_model=ScanResult)
    def scan(facts: PostureFacts, use_model: bool = False) -> ScanResult:
        result = scorer.score(facts, feed)
        if use_model and risk_model is not None:
            model_score, explanation = risk_model.predict(featurize(facts))
            result = result.model_copy(update={"score": model_score, "score_explanation": explanation})
        training_logger.record(facts, result)
        return result

    return router
