"""Shared fixtures for the paperscope test suite.

Offline-suite guard: the entire suite is offline-by-design — every network
interaction is mocked at the requests layer (see tests/test_openalex_client.py
for the pattern). The autouse fixture below makes that a *checked invariant*:
any test that tries to open a real network connection (or resolve a non-local
hostname) fails immediately with NetworkAccessBlocked instead of silently
hitting — or hanging on — the network.

Self-tests for this guard live in tests/test_cli_smoke.py
(TestOfflineNetworkGuard).

Currently NO test needs a live socket, so nothing is allowlisted — not even
localhost. If a future test genuinely needs a local socket, add its host to
ALLOWED_HOSTS below (e.g. "127.0.0.1", "::1", "localhost") and nothing else.
"""

from __future__ import annotations

import socket

import pytest

# Hosts tests may connect to / resolve. Deliberately empty: the suite is
# fully offline. Extend only for a genuine local-socket need.
ALLOWED_HOSTS = frozenset()


class NetworkAccessBlocked(RuntimeError):
    """A test attempted real network access in the offline-by-design suite."""


def _blocked_message(operation: str, target: object) -> str:
    return (
        f"offline test suite: blocked {operation} to {target!r}. "
        f"paperscope tests must not touch the network — mock HTTP at the "
        f"requests layer (see tests/test_openalex_client.py) or, if a test "
        f"genuinely needs a local socket, add the host to ALLOWED_HOSTS in "
        f"tests/conftest.py."
    )


def _host_allowed(host: object) -> bool:
    return isinstance(host, str) and host in ALLOWED_HOSTS


def _check_connect_address(address: object) -> None:
    # AF_UNIX addresses are filesystem paths (str/bytes): local IPC, allowed.
    if isinstance(address, (str, bytes)):
        return
    host = address[0] if isinstance(address, tuple) and address else None
    if not _host_allowed(host):
        raise NetworkAccessBlocked(_blocked_message("socket connect", address))


@pytest.fixture(autouse=True)
def _no_network_access():
    """Autouse guard: block socket connects and DNS resolution for all tests."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, address):
        _check_connect_address(address)
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        _check_connect_address(address)
        return real_connect_ex(self, address)

    def guarded_getaddrinfo(host, port, *args, **kwargs):
        if not _host_allowed(host):
            raise NetworkAccessBlocked(
                _blocked_message("getaddrinfo", (host, port))
            )
        return real_getaddrinfo(host, port, *args, **kwargs)

    # `connect`/`connect_ex` are inherited from the C base class, so restore
    # by deleting the shadowing attribute rather than re-assigning it.
    had_connect = "connect" in socket.socket.__dict__
    had_connect_ex = "connect_ex" in socket.socket.__dict__
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = guarded_getaddrinfo
    try:
        yield
    finally:
        if had_connect:
            socket.socket.connect = real_connect
        else:
            del socket.socket.connect
        if had_connect_ex:
            socket.socket.connect_ex = real_connect_ex
        else:
            del socket.socket.connect_ex
        socket.getaddrinfo = real_getaddrinfo
