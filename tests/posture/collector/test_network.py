from collections import namedtuple

import psutil

from riskvane.posture.collector import network

_Addr = namedtuple("Addr", ["ip", "port"])
_Conn = namedtuple("Conn", ["laddr", "status", "pid", "type"])


class _FakeProcess:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


def test_collects_listening_ports_with_process_name(monkeypatch):
    conns = [
        _Conn(laddr=_Addr("0.0.0.0", 22), status=psutil.CONN_LISTEN, pid=100, type=1),
        _Conn(laddr=_Addr("127.0.0.1", 5000), status=psutil.CONN_ESTABLISHED, pid=200, type=1),
    ]
    monkeypatch.setattr(psutil, "Process", lambda pid: _FakeProcess("sshd"))

    result = network.collect(connections=lambda kind: conns)

    assert len(result) == 1
    assert result[0].port == 22
    assert result[0].protocol == "tcp"
    assert result[0].listen_address == "0.0.0.0"
    assert result[0].process_name == "sshd"


def test_handles_process_lookup_failure(monkeypatch):
    conns = [_Conn(laddr=_Addr("0.0.0.0", 6379), status=psutil.CONN_LISTEN, pid=999, type=1)]

    def raise_no_such_process(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", raise_no_such_process)

    result = network.collect(connections=lambda kind: conns)

    assert result[0].process_name is None
