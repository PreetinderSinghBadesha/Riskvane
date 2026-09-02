"""Process and user facts: totals, root-owned processes, sudo/wheel group members."""

from __future__ import annotations

import psutil

from riskvane.common.schema import ProcessSummary

_SUDO_GROUPS = ("sudo", "wheel")


def _default_getgrnam(name: str):
    import grp  # POSIX-only; imported lazily so this module loads on any platform

    return grp.getgrnam(name)


def _sudo_users(getgrnam=_default_getgrnam) -> list[str]:
    members: set[str] = set()
    for group_name in _SUDO_GROUPS:
        try:
            members.update(getgrnam(group_name).gr_mem)
        except (KeyError, ImportError):
            continue
    return sorted(members)


def collect(process_iter=psutil.process_iter, getgrnam=_default_getgrnam) -> ProcessSummary:
    total = 0
    root_count = 0
    for proc in process_iter(["username"]):
        total += 1
        try:
            if proc.info.get("username") == "root":
                root_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return ProcessSummary(
        total_processes=total,
        root_processes=root_count,
        sudo_users=_sudo_users(getgrnam),
    )
