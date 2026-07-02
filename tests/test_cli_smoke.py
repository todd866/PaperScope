"""Parse-only smoke tests over BOTH CLI dispatchers.

Covers ~1,550 LOC of otherwise-untested argparse wiring:

- ``python3 -m paperscope``                    (paperscope/__main__.py)
- ``python3 -m paperscope.systematic_review``  (systematic_review/__main__.py)

Subcommands are enumerated DYNAMICALLY from the parser objects (including
nested subcommands such as ``validate workbook``), so anything added to
either dispatcher is covered automatically — no hardcoded command list.
For dispatchers without a ``build_parser()`` entry point, the parser is
captured by intercepting ``ArgumentParser.parse_args`` during ``main()``.

Lazy dispatch-target imports are enumerated from the dispatcher source via
``ast`` and resolved with ``importlib``, catching import regressions (moved
modules, renamed functions) that ``--help`` alone would not.

Also holds the self-tests for the offline-suite socket guard installed by
tests/conftest.py (the "no test hits the network" checked invariant).
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import socket
import sys
from pathlib import Path

import pytest

MAIN_DISPATCHER = "paperscope.__main__"
SR_DISPATCHER = "paperscope.systematic_review.__main__"
DISPATCHERS = (MAIN_DISPATCHER, SR_DISPATCHER)


# ---------------------------------------------------------------------------
# Parser capture + dynamic subcommand enumeration
# ---------------------------------------------------------------------------


class _ParserCaptured(Exception):
    def __init__(self, parser):
        self.parser = parser


def _build_dispatcher_parser(dispatcher: str) -> argparse.ArgumentParser:
    """Get the dispatcher's fully-wired parser without dispatching anything.

    Prefers a ``build_parser()`` entry point when the module exposes one;
    otherwise runs ``main()`` with ``ArgumentParser.parse_args`` intercepted
    so we capture the exact parser object the dispatcher builds.
    """
    module = importlib.import_module(dispatcher)
    if hasattr(module, "build_parser"):
        return module.build_parser()

    original = argparse.ArgumentParser.parse_args

    def _spy(self, args=None, namespace=None):
        raise _ParserCaptured(self)

    argparse.ArgumentParser.parse_args = _spy
    try:
        module.main()
    except _ParserCaptured as captured:
        return captured.parser
    finally:
        argparse.ArgumentParser.parse_args = original
    raise AssertionError(f"{dispatcher}.main() never called parse_args()")


_PARSER_CACHE = {}


def _parser(dispatcher: str) -> argparse.ArgumentParser:
    if dispatcher not in _PARSER_CACHE:
        _PARSER_CACHE[dispatcher] = _build_dispatcher_parser(dispatcher)
    return _PARSER_CACHE[dispatcher]


def _walk_subcommands(parser, path=()):
    """Yield ``(path_tuple, subparser)`` for every (nested) subcommand."""
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        seen = set()
        for name, sub in action.choices.items():
            if id(sub) in seen:  # alias for an already-yielded subparser
                continue
            seen.add(id(sub))
            sub_path = path + (name,)
            yield sub_path, sub
            for nested in _walk_subcommands(sub, sub_path):
                yield nested


def _command_cases():
    cases = []
    for dispatcher in DISPATCHERS:
        for path, _sub in _walk_subcommands(_parser(dispatcher)):
            cases.append(pytest.param(dispatcher, path, id=f"{dispatcher}:{' '.join(path)}"))
    return cases


def _is_leaf(subparser) -> bool:
    return not any(
        isinstance(a, argparse._SubParsersAction) for a in subparser._actions
    )


# ---------------------------------------------------------------------------
# Every subcommand parses (--help exits 0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dispatcher, path", _command_cases())
def test_every_subcommand_help_parses_and_exits_zero(dispatcher, path, capsys):
    parser = _parser(dispatcher)
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(list(path) + ["--help"])
    assert excinfo.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("dispatcher", DISPATCHERS)
def test_dispatchers_expose_at_least_one_subcommand(dispatcher):
    assert list(_walk_subcommands(_parser(dispatcher)))


def test_main_dispatcher_core_subcommands_present():
    # Floor set only (new subcommands are picked up dynamically above).
    names = {path[0] for path, _ in _walk_subcommands(_parser(MAIN_DISPATCHER))}
    assert {"extract", "resolve", "verify", "analyze", "forensic"} <= names


def test_sr_dispatcher_core_subcommands_present():
    names = {path[0] for path, _ in _walk_subcommands(_parser(SR_DISPATCHER))}
    assert {"aggregate", "prisma", "search", "build-site", "validate"} <= names


# ---------------------------------------------------------------------------
# Minimal plausible argv for a few stable subcommands (namespace sanity)
# ---------------------------------------------------------------------------


def test_main_extract_parses_minimal_argv():
    ns = _parser(MAIN_DISPATCHER).parse_args(["extract", "/tmp/proj", "--stats-only"])
    assert ns.command == "extract"
    assert ns.project_root == Path("/tmp/proj")
    assert ns.stats_only is True


def test_main_forensic_parses_minimal_argv():
    ns = _parser(MAIN_DISPATCHER).parse_args(["forensic", "/tmp/tables.json"])
    assert ns.command == "forensic"
    assert ns.input == Path("/tmp/tables.json")


def test_sr_prisma_parses_minimal_argv():
    ns = _parser(SR_DISPATCHER).parse_args(["prisma", "--corpus", "/tmp/corpus"])
    assert ns.corpus == "/tmp/corpus"
    assert callable(ns.fn)


def test_sr_leaf_subcommands_all_wire_a_dispatch_fn():
    """Every leaf subcommand of the SR dispatcher must set_defaults(fn=...).

    ``main()`` dispatches via ``args.fn(args)``; a leaf without an ``fn``
    default would crash with AttributeError at dispatch time.
    """
    missing = [
        " ".join(path)
        for path, sub in _walk_subcommands(_parser(SR_DISPATCHER))
        if _is_leaf(sub) and not callable(sub._defaults.get("fn"))
    ]
    assert not missing, f"SR leaf subcommands without a dispatch fn: {missing}"


# ---------------------------------------------------------------------------
# No-args / unknown-command behaviour
# ---------------------------------------------------------------------------


def test_main_no_args_prints_help_and_returns_1(monkeypatch, capsys):
    module = importlib.import_module(MAIN_DISPATCHER)
    monkeypatch.setattr(sys, "argv", ["paperscope"])
    assert module.main() == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_sr_no_args_exits_2(capsys):
    module = importlib.import_module(SR_DISPATCHER)
    with pytest.raises(SystemExit) as excinfo:
        module.main([])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("dispatcher", DISPATCHERS)
def test_unknown_subcommand_exits_2(dispatcher, capsys):
    with pytest.raises(SystemExit) as excinfo:
        _parser(dispatcher).parse_args(["definitely-not-a-real-subcommand"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Lazy dispatch-target imports resolve
# ---------------------------------------------------------------------------


def _lazy_import_targets(dispatcher: str):
    """(module, imported-names) pairs for every paperscope import — module
    level or lazy inside a command handler — in the dispatcher's source."""
    module = importlib.import_module(dispatcher)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    package = module.__package__
    targets = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), package
                )
            else:
                base = node.module or ""
            if base.split(".")[0] == "paperscope":
                targets.setdefault(base, set()).update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "paperscope":
                    targets.setdefault(alias.name, set())
    return sorted((base, sorted(names)) for base, names in targets.items())


def _import_cases():
    cases = []
    for dispatcher in DISPATCHERS:
        for base, names in _lazy_import_targets(dispatcher):
            cases.append(pytest.param(dispatcher, base, names, id=f"{dispatcher}->{base}"))
    return cases


@pytest.mark.parametrize("dispatcher, base, names", _import_cases())
def test_dispatch_target_imports_resolve(dispatcher, base, names):
    try:
        module = importlib.import_module(base)
    except ImportError as exc:
        root = (getattr(exc, "name", None) or "").split(".")[0]
        if root and root != "paperscope":
            pytest.skip(
                f"third-party dependency {exc.name!r} (needed by {base}) "
                f"not installed in this environment"
            )
        raise
    missing = []
    for name in names:
        if hasattr(module, name):
            continue
        try:  # `from pkg import submodule` style
            if importlib.util.find_spec(f"{base}.{name}") is not None:
                continue
        except (ImportError, ModuleNotFoundError, AttributeError):
            pass
        missing.append(name)
    assert not missing, (
        f"{dispatcher} imports {missing} from {base}, but {base} does not "
        f"provide them — dispatch would crash with ImportError"
    )


# ---------------------------------------------------------------------------
# Offline-suite socket guard (installed by tests/conftest.py)
# ---------------------------------------------------------------------------


class TestOfflineNetworkGuard:
    """The suite is offline-by-design; conftest.py makes that a checked
    invariant. These self-tests prove the guard is actually armed.

    Timeouts are set so that, should the guard ever be missing, the tests
    fail fast (with a non-RuntimeError) instead of hanging on real I/O.
    """

    def test_external_tcp_connect_blocked(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(0.5)
            with pytest.raises(RuntimeError, match="offline"):
                s.connect(("203.0.113.1", 80))  # TEST-NET-3, never routable
        finally:
            s.close()

    def test_external_connect_ex_blocked(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(0.5)
            with pytest.raises(RuntimeError, match="offline"):
                s.connect_ex(("203.0.113.1", 80))
        finally:
            s.close()

    def test_dns_resolution_blocked(self):
        with pytest.raises(RuntimeError, match="offline"):
            socket.getaddrinfo("paperscope-guard-check.invalid", 443)

    def test_localhost_also_blocked(self):
        # No test in this suite needs a live local socket (all network I/O is
        # mocked at the requests layer), so the guard blocks localhost too.
        # If a future test genuinely needs one, extend ALLOWED_HOSTS in
        # tests/conftest.py.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(0.5)
            with pytest.raises(RuntimeError, match="offline"):
                s.connect(("127.0.0.1", 9))
        finally:
            s.close()

    def test_socket_creation_itself_still_allowed(self):
        # Only connection establishment is guarded; constructing sockets
        # (e.g. for ephemeral-port discovery patterns) must keep working.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.close()
