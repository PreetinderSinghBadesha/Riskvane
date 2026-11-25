"""Unified Riskvane service: posture scanning + live attack detection behind one API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from riskvane.detection.api.app import create_router as create_detection_router
from riskvane.posture.api.app import create_router as create_posture_router
from riskvane.rl.api import create_router as create_remediation_router


def create_app(
    training_log_path: Path | None = None,
    model_path: Path | None = None,
    features_path: Path | None = None,
    q_table_path: Path | None = None,
    dqn_model_path: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Riskvane")
    app.include_router(
        create_posture_router(
            training_log_path=training_log_path,
            model_path=model_path,
            features_path=features_path,
        )
    )
    app.include_router(create_detection_router())
    app.include_router(
        create_remediation_router(q_table_path=q_table_path, dqn_model_path=dqn_model_path)
    )
    return app
