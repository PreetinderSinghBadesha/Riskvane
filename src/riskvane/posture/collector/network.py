"""Open port facts: port, protocol, listen address, owning process."""

from __future__ import annotations

import psutil

from riskvane.common.schema import OpenPort


def _proto_name(kind_or_type: int) -> str:
    # socket.SOCK_STREAM == 1 -> tcp, socket.SOCK_DGRAM == 2 -> udp
    return "tcp" if kind_or_type == 1 else "udp"


def collect(connections=psutil.net_connections) -> list[OpenPort]:
    ports: list[OpenPort] = []
    for conn in connections(kind="inet"):
        if conn.status != psutil.CONN_LISTEN:
            continue
        process_name = None
        if conn.pid:
            try:
                process_name = psutil.Process(conn.pid).name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                process_name = None
        ports.append(
            OpenPort(
                port=conn.laddr.port,
                protocol=_proto_name(conn.type),
                listen_address=conn.laddr.ip,
                pid=conn.pid,
                process_name=process_name,
            )
        )
    return ports
