"""Installed package facts, via whichever package manager is present."""

from __future__ import annotations

import shutil
import subprocess

from riskvane.common.schema import Package


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.stdout


def _parse_name_version_lines(output: str) -> list[Package]:
    packages: list[Package] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or " " not in line:
            continue
        name, _, version = line.partition(" ")
        packages.append(Package(name=name, version=version))
    return packages


def collect(run=_run, which=shutil.which) -> list[Package]:
    if which("dpkg-query"):
        output = run(["dpkg-query", "-W", "-f=${Package} ${Version}\n"])
        return _parse_name_version_lines(output)
    if which("rpm"):
        output = run(["rpm", "-qa", "--qf", "%{NAME} %{VERSION}-%{RELEASE}\n"])
        return _parse_name_version_lines(output)
    return []
