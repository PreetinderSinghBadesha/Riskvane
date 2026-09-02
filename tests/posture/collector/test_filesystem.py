import os
import stat

from riskvane.posture.collector import filesystem


class _FakeStat:
    def __init__(self, st_mode):
        self.st_mode = st_mode


def test_flags_world_writable_and_suid_files(tmp_path, monkeypatch):
    root = tmp_path / "scan_root"
    root.mkdir()
    normal = root / "normal.txt"
    normal.write_text("hi")
    writable = root / "writable.txt"
    writable.write_text("hi")
    suid_bin = root / "suid_bin"
    suid_bin.write_text("hi")

    def fake_lstat(path):
        if path == str(writable):
            return _FakeStat(stat.S_IFREG | stat.S_IWOTH | 0o644)
        if path == str(suid_bin):
            return _FakeStat(stat.S_IFREG | stat.S_ISUID | 0o755)
        return _FakeStat(stat.S_IFREG | 0o644)

    monkeypatch.setattr(os, "lstat", fake_lstat)

    findings = filesystem.collect(roots=[str(root)])

    issues = {(f.path, f.issue) for f in findings}
    assert (str(writable), "world_writable") in issues
    assert (str(suid_bin), "suid") in issues
    assert not any(f.path == str(normal) for f in findings)


def test_respects_max_files_budget(tmp_path, monkeypatch):
    root = tmp_path / "scan_root"
    root.mkdir()
    for i in range(5):
        (root / f"file{i}.txt").write_text("hi")

    monkeypatch.setattr(
        os, "lstat", lambda path: _FakeStat(stat.S_IFREG | stat.S_IWOTH | 0o646)
    )

    findings = filesystem.collect(roots=[str(root)], max_files=1)

    assert len(findings) == 1


def test_skips_missing_root():
    findings = filesystem.collect(roots=[str(_nonexistent_path())])

    assert findings == []


def _nonexistent_path():
    return os.path.join(os.path.dirname(__file__), "definitely-does-not-exist")
