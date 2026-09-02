"""v0 posture API: POST /scan."""

from __future__ import annotations

from fastapi import FastAPI

from riskvane.common.schema import PostureFacts, ScanResult
from riskvane.posture.scoring import cve_feed, scorer


def create_app() -> FastAPI:
    app = FastAPI(title="Riskvane Posture API")
    feed = cve_feed.load_feed()

    @app.post("/scan", response_model=ScanResult)
    def scan(facts: PostureFacts) -> ScanResult:
        return scorer.score(facts, feed)

    return app
