"""OS and kernel version facts."""

from __future__ import annotations

import platform

from riskvane.common.schema import OsInfo

_OS_RELEASE_PATH = "/etc/os-release"


def _parse_os_release(path: str = _OS_RELEASE_PATH) -> dict[str, str]:
    fields: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                fields[key] = value.strip().strip('"')
    except FileNotFoundError:
        pass
    return fields


def collect(os_release_path: str = _OS_RELEASE_PATH) -> OsInfo:
    release = _parse_os_release(os_release_path)
    name = release.get("PRETTY_NAME") or release.get("NAME") or platform.system()
    version = release.get("VERSION") or release.get("VERSION_ID") or platform.release()
    return OsInfo(
        name=name,
        version=version,
        kernel=platform.release(),
        architecture=platform.machine(),
    )
