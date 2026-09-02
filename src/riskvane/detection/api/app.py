"""v0 log-stream API: POST /events, GET /alerts?vm_id=."""

from __future__ import annotations

from fastapi import APIRouter

from riskvane.common.events import Alert, Event
from riskvane.detection.engine import DetectionEngine


def create_router() -> APIRouter:
    router = APIRouter()
    engine = DetectionEngine()

    @router.post("/events", response_model=list[Alert])
    def ingest_events(events: list[Event]) -> list[Alert]:
        fired: list[Alert] = []
        for event in events:
            fired.extend(engine.process_event(event))
        return fired

    @router.get("/alerts", response_model=list[Alert])
    def get_alerts(vm_id: str) -> list[Alert]:
        return engine.get_alerts(vm_id)

    return router
