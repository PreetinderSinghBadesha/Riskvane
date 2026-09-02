"""Best-effort version-string comparison for distro package versions.

Package versions from dpkg/rpm aren't uniform ("5.1-6ubuntu1", "7.76.1-23.el9"), and OpenSSL-style
versions add a single trailing patch letter ("1.1.1n"). Rather than depend on a full Debian/RPM
version-comparison algorithm, this extracts the leading dotted-numeric prefix plus that optional
trailing letter, and compares the result as a tuple. Good enough to decide "installed < fixed_in"
for the seed CVE feed's version strings; a real feed integration may need a proper algorithm later.
"""

from __future__ import annotations

import re

_VERSION_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)([a-zA-Z])?")


def parse(version: str) -> tuple[int, ...]:
    match = _VERSION_PATTERN.match(version.strip())
    if not match:
        return ()
    numeric_part, letter = match.group(1), match.group(2)
    parts = tuple(int(p) for p in numeric_part.split("."))
    if letter:
        parts += (ord(letter.lower()),)
    return parts


def is_older_than(installed: str, fixed_in: str) -> bool:
    installed_parts = parse(installed)
    fixed_parts = parse(fixed_in)
    if not installed_parts or not fixed_parts:
        return False
    return installed_parts < fixed_parts
