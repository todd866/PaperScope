"""Packaging invariants: version single-sourcing, Python floor, metadata sync.

Self-contained — reads repo files only, no network.

  python tests/test_packaging.py    # standalone, prints PASS/FAIL
  pytest tests/test_packaging.py    # under pytest
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from paperscope import __version__  # noqa: E402

# The empirically-determined floor: PEP 604/585 annotations are guarded by
# `from __future__ import annotations`, and the full suite passes on 3.9.
PYTHON_FLOOR = "3.9"

_ANNOTATION_BUILTINS = {"list", "dict", "tuple", "set", "frozenset", "type"}


def _annotations_of(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            yield node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for a in [*args.args, *args.posonlyargs, *args.kwonlyargs, args.vararg, args.kwarg]:
                if a is not None and a.annotation is not None:
                    yield a.annotation
            if node.returns is not None:
                yield node.returns


def _uses_modern_annotation(annotation: ast.AST) -> bool:
    """PEP 604 unions (X | Y) or PEP 585 builtin generics (list[str])."""
    for n in ast.walk(annotation):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr):
            return True
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id in _ANNOTATION_BUILTINS):
            return True
    return False


def test_future_import_guards_modern_annotations():
    """Any module using PEP 604/585 annotation syntax must carry the future
    import, or it crashes at import time on Python < 3.10 (annotations are
    evaluated eagerly without it — including module-level AnnAssign)."""
    offenders = []
    for path in sorted((ROOT / "paperscope").rglob("*.py")):
        src = path.read_text(errors="ignore")
        if "from __future__ import annotations" in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue  # compileall / CI catches real syntax errors
        if any(_uses_modern_annotation(a) for a in _annotations_of(tree)):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        "modules use PEP 604/585 annotations without "
        f"'from __future__ import annotations': {offenders}"
    )


def test_pyproject_exists_and_declares_floor():
    pyproject = ROOT / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml missing"
    text = pyproject.read_text()
    m = re.search(r'requires-python\s*=\s*">=([\d.]+)"', text)
    assert m, "requires-python not declared in pyproject.toml"
    assert m.group(1) == PYTHON_FLOOR, (
        f"declared floor {m.group(1)} != empirical floor {PYTHON_FLOOR}"
    )
    # Version must be dynamic (single-sourced from paperscope/__init__.py).
    assert re.search(r'dynamic\s*=\s*\[[^\]]*"version"', text), (
        "version must be dynamic (attr = paperscope.__version__)"
    )


def test_plugin_json_version_matches_package():
    """plugin.json can't read __version__ dynamically; a release must keep
    them in lockstep."""
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    assert plugin["version"] == __version__, (
        f"plugin.json version {plugin['version']} != paperscope.__version__ {__version__}"
    )


def test_plugin_json_repository_url():
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    assert plugin["repository"] == "https://github.com/todd866/PaperScope"
    assert plugin["homepage"] == "https://github.com/todd866/PaperScope"


def test_arxiv_user_agent_single_sources_version():
    """The arXiv User-Agent must be built from __version__, not hardcoded."""
    from paperscope.harvest.sources.arxiv import ArxivSource
    ua = ArxivSource().session.headers["User-Agent"]
    assert f"paperscope/{__version__}" in ua, f"User-Agent {ua!r} not built from __version__"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    raise SystemExit(1 if failures else 0)
