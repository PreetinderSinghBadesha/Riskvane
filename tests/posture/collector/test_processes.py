from riskvane.posture.collector import processes


class _FakeProc:
    def __init__(self, username):
        self.info = {"username": username}


class _FakeGroup:
    def __init__(self, gr_mem):
        self.gr_mem = gr_mem


def test_counts_total_and_root_processes():
    procs = [_FakeProc("root"), _FakeProc("alice"), _FakeProc("root")]

    def getgrnam(name):
        raise KeyError(name)

    result = processes.collect(process_iter=lambda fields: procs, getgrnam=getgrnam)

    assert result.total_processes == 3
    assert result.root_processes == 2
    assert result.sudo_users == []


def test_collects_sudo_and_wheel_members():
    groups = {"sudo": _FakeGroup(["alice", "bob"]), "wheel": _FakeGroup(["bob", "carol"])}

    def getgrnam(name):
        if name not in groups:
            raise KeyError(name)
        return groups[name]

    result = processes.collect(process_iter=lambda fields: [], getgrnam=getgrnam)

    assert result.sudo_users == ["alice", "bob", "carol"]
