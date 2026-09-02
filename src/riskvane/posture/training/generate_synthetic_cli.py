"""Entrypoint: riskvane-generate-synthetic-corpus [--out FILE]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from riskvane.posture.scoring import cve_feed, scorer
from riskvane.posture.training.logger import TrainingDataLogger
from riskvane.posture.training.synthetic import generate_facts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="riskvane-generate-synthetic-corpus",
        description="Generate the synthetic misconfig corpus and append it to the training-data JSONL.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Training-data JSONL path (default: var/training/posture_scans.jsonl).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    feed = cve_feed.load_feed()
    logger = TrainingDataLogger(Path(args.out)) if args.out else TrainingDataLogger()

    count = 0
    for facts in generate_facts():
        result = scorer.score(facts, feed)
        logger.record(facts, result, source="synthetic")
        count += 1

    print(f"Generated and logged {count} synthetic training examples.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
