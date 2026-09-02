"""v0 posture API: POST /scan."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from riskvane.common.schema import PostureFacts, ScanResult
from riskvane.posture.scoring import cve_feed, scorer
from riskvane.posture.training.logger import TrainingDataLogger


def create_router(training_log_path: Path | None = None) -> APIRouter:
    router = APIRouter()
    feed = cve_feed.load_feed()
    training_logger = (
        TrainingDataLogger(training_log_path) if training_log_path else TrainingDataLogger()
    )

    @router.post("/scan", response_model=ScanResult)
    def scan(facts: PostureFacts) -> ScanResult:
        result = scorer.score(facts, feed)
        training_logger.record(facts, result)
        return result

    return router
