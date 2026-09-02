"""Unified Riskvane service: posture scanning + live attack detection behind one API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from riskvane.detection.api.app import create_router as create_detection_router
from riskvane.posture.api.app import create_router as create_posture_router


def create_app(training_log_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="Riskvane")
    app.include_router(create_posture_router(training_log_path=training_log_path))
    app.include_router(create_detection_router())
    return app
