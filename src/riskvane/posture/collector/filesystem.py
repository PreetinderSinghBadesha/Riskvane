"""World-writable file and SUID binary facts, bounded by a scan budget."""

from __future__ import annotations

import os
import stat
import time

from riskvane.common.schema import FilesystemFinding

_DEFAULT_ROOTS = ["/usr", "/bin", "/sbin", "/opt", "/home"]
_EXCLUDED_DIRS = {"/proc", "/sys"}
_MAX_FILES_SCANNED = 200_000
_MAX_SECONDS = 30.0


def _is_world_writable(mode: int) -> bool:
    return bool(mode & stat.S_IWOTH)


def _is_suid(mode: int) -> bool:
    return bool(mode & stat.S_ISUID)


def collect(
    roots: list[str] | None = None,
    max_files: int = _MAX_FILES_SCANNED,
    max_seconds: float = _MAX_SECONDS,
) -> list[FilesystemFinding]:
    findings: list[FilesystemFinding] = []
    scanned = 0
    start = time.monotonic()

    for root in roots if roots is not None else _DEFAULT_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if os.path.join(dirpath, d) not in _EXCLUDED_DIRS]
            for filename in filenames:
                if scanned >= max_files or (time.monotonic() - start) >= max_seconds:
                    return findings
                scanned += 1
                path = os.path.join(dirpath, filename)
                try:
                    mode = os.lstat(path).st_mode
                except OSError:
                    continue
                if not stat.S_ISREG(mode):
                    continue
                if _is_world_writable(mode):
                    findings.append(FilesystemFinding(path=path, issue="world_writable"))
                if _is_suid(mode):
                    findings.append(FilesystemFinding(path=path, issue="suid"))
    return findings
