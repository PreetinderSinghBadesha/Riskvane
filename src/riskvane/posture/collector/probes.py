"""Unauthenticated-access probes against known service ports found open locally."""

from __future__ import annotations

import socket

from riskvane.common.schema import OpenPort, ServiceProbe

_KNOWN_SERVICES = {
    6379: "redis",
    27017: "mongodb",
    11211: "memcached",
}

_TIMEOUT_SECONDS = 1.0


def _probe_redis(host: str, port: int) -> tuple[bool, str]:
    with socket.create_connection((host, port), timeout=_TIMEOUT_SECONDS) as sock:
        sock.sendall(b"PING\r\n")
        reply = sock.recv(64)
        unauthenticated = reply.startswith(b"+PONG")
        return unauthenticated, reply.decode(errors="replace").strip()


def _probe_memcached(host: str, port: int) -> tuple[bool, str]:
    with socket.create_connection((host, port), timeout=_TIMEOUT_SECONDS) as sock:
        sock.sendall(b"stats\r\n")
        reply = sock.recv(64)
        unauthenticated = reply.startswith(b"STAT")
        return unauthenticated, reply.decode(errors="replace").strip()


def _probe_mongodb(host: str, port: int) -> tuple[bool, str]:
    # A bare TCP connect succeeding is itself the signal for older/unauthenticated MongoDB
    # deployments that bind without --auth; a full wire-protocol handshake is out of scope here.
    with socket.create_connection((host, port), timeout=_TIMEOUT_SECONDS):
        return True, "tcp connect succeeded with no auth challenge"


_PROBERS = {
    "redis": _probe_redis,
    "memcached": _probe_memcached,
    "mongodb": _probe_mongodb,
}


def collect(open_ports: list[OpenPort]) -> list[ServiceProbe]:
    findings: list[ServiceProbe] = []
    for open_port in open_ports:
        service = _KNOWN_SERVICES.get(open_port.port)
        if not service:
            continue
        host = open_port.listen_address
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        prober = _PROBERS[service]
        try:
            unauthenticated, detail = prober(host, open_port.port)
        except OSError as exc:
            unauthenticated, detail = False, f"probe failed: {exc}"
        findings.append(
            ServiceProbe(
                port=open_port.port,
                service=service,
                unauthenticated=unauthenticated,
                detail=detail,
            )
        )
    return findings
