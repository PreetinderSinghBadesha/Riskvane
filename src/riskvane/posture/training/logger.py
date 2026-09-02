"""Appends (features, label) pairs from real and synthetic scans to a local JSONL training corpus."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from riskvane.common.schema import PostureFacts, ScanResult
from riskvane.posture.training.features import featurize

_DEFAULT_PATH = Path("var/training/posture_scans.jsonl")


class TrainingDataLogger:
    def __init__(self, path: Path = _DEFAULT_PATH):
        self._path = path

    def record(
        self,
        facts: PostureFacts,
        result: ScanResult,
        source: Literal["live", "synthetic"] = "live",
    ) -> None:
        row = {
            "vm_id": facts.vm_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "features": featurize(facts),
            "label": result.score,
            "severity_breakdown": result.severity_breakdown,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
