import json

from riskvane.common.schema import OsInfo, PostureFacts, ProcessSummary, ScanResult, SshConfig
from riskvane.posture.training.logger import TrainingDataLogger


def _facts() -> PostureFacts:
    return PostureFacts(
        vm_id="test-vm",
        os=OsInfo(name="Ubuntu", version="22.04", kernel="5.15", architecture="x86_64"),
        ssh=SshConfig(),
        processes=ProcessSummary(total_processes=10, root_processes=1),
    )


def _result(score: int = 42) -> ScanResult:
    breakdown = {"critical": 0, "high": 1, "medium": 0, "low": 0, "info": 0}
    return ScanResult(score=score, severity_breakdown=breakdown, findings=[])


def test_record_writes_one_well_formed_json_line(tmp_path):
    path = tmp_path / "training" / "posture_scans.jsonl"
    logger = TrainingDataLogger(path)

    logger.record(_facts(), _result(score=42))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["vm_id"] == "test-vm"
    assert row["label"] == 42
    assert "num_packages" in row["features"]
    assert row["severity_breakdown"]["high"] == 1
    assert "recorded_at" in row
    assert row["source"] == "live"


def test_record_tags_synthetic_source(tmp_path):
    path = tmp_path / "posture_scans.jsonl"
    logger = TrainingDataLogger(path)

    logger.record(_facts(), _result(), source="synthetic")

    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["source"] == "synthetic"


def test_record_appends_subsequent_rows(tmp_path):
    path = tmp_path / "posture_scans.jsonl"
    logger = TrainingDataLogger(path)

    logger.record(_facts(), _result(score=10))
    logger.record(_facts(), _result(score=20))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["label"] == 10
    assert json.loads(lines[1])["label"] == 20


def test_record_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dirs" / "posture_scans.jsonl"
    logger = TrainingDataLogger(path)

    logger.record(_facts(), _result())

    assert path.exists()
