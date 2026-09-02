"""sshd_config facts: root login, password auth, protocol."""

from __future__ import annotations

import glob
import os

from riskvane.common.schema import SshConfig

_SSHD_CONFIG_PATH = "/etc/ssh/sshd_config"
_TRACKED_KEYS = {
    "permitrootlogin": "permit_root_login",
    "passwordauthentication": "password_authentication",
    "protocol": "protocol",
}


def _iter_lines(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            yield from f
    except FileNotFoundError:
        return


def _parse(path: str, values: dict[str, str], base_dir: str) -> None:
    for line in _iter_lines(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts[0].lower(), parts[1].strip()
        if key == "include":
            for included in glob.glob(os.path.join(base_dir, value)):
                _parse(included, values, base_dir)
            continue
        mapped = _TRACKED_KEYS.get(key)
        if mapped and mapped not in values:
            values[mapped] = value


def collect(sshd_config_path: str = _SSHD_CONFIG_PATH) -> SshConfig:
    values: dict[str, str] = {}
    _parse(sshd_config_path, values, os.path.dirname(sshd_config_path) or "/etc/ssh")
    return SshConfig(**values)
