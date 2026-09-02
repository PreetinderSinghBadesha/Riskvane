"""Entrypoint: riskvane-collect [--vm-id ID] [--out FILE]"""

from __future__ import annotations

import argparse
import socket
import sys

from riskvane.posture.collector import facts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="riskvane-collect",
        description="Collect raw VM posture facts and emit them as JSON.",
    )
    parser.add_argument(
        "--vm-id",
        default=None,
        help="Identifier for this VM (defaults to the local hostname).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional file path to also write the JSON payload to.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    vm_id = args.vm_id or socket.gethostname()

    payload = facts.collect(vm_id).model_dump_json(indent=2)
    print(payload)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)

    return 0


if __name__ == "__main__":
    sys.exit(main())
