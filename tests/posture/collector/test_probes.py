from riskvane.common.schema import OpenPort
from riskvane.posture.collector import probes


def test_ignores_ports_without_a_known_service():
    ports = [OpenPort(port=8080, protocol="tcp", listen_address="0.0.0.0")]

    result = probes.collect(ports)

    assert result == []


def test_flags_unauthenticated_redis(monkeypatch):
    ports = [OpenPort(port=6379, protocol="tcp", listen_address="0.0.0.0")]
    monkeypatch.setitem(probes._PROBERS, "redis", lambda host, port: (True, "+PONG"))

    result = probes.collect(ports)

    assert len(result) == 1
    assert result[0].service == "redis"
    assert result[0].unauthenticated is True


def test_probe_failure_is_reported_not_raised(monkeypatch):
    ports = [OpenPort(port=27017, protocol="tcp", listen_address="0.0.0.0")]

    def raise_connection_refused(host, port):
        raise OSError("connection refused")

    monkeypatch.setitem(probes._PROBERS, "mongodb", raise_connection_refused)

    result = probes.collect(ports)

    assert result[0].unauthenticated is False
    assert "connection refused" in result[0].detail
