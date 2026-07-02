"""Documentation-drift guards.

These tests fail when the docs fall out of sync with the shipped code:

(a) every subcommand of both CLI dispatchers is mentioned in README.md
(b) every function named in the README forensic reference table exists in
    paperscope.analysis.forensic_stats
(c) every relative markdown link in README / CLAUDE / ARCHITECTURE / docs
    resolves to a file (or directory) that exists
(d) the README Python-badge floor matches pyproject's requires-python

All assertions are repo-relative (anchored at the repository root inferred
from this file's location), so the module can sync unchanged to the public
mirror.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"


# ─────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────
def _dispatcher_subcommands(module: str) -> list[str]:
    """Subcommand names of an argparse dispatcher, read from its --help.

    argparse renders the choices as ``{a,b,c}`` in the usage/positional
    block; we take the content of the first brace group and split it,
    tolerating line wrapping inside the group.
    """
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    text = proc.stdout or proc.stderr
    start = text.index("{")
    end = text.index("}", start)
    raw = text[start + 1 : end].replace("\n", " ")
    return [s.strip() for s in raw.split(",") if s.strip()]


def _forensic_reference_functions() -> list[str]:
    """Function names in the README 'Forensic Statistics Reference' table.

    Only the leading ``| `name()` |`` cell of each table row is taken, so
    functions merely mentioned in prose are not required to appear here.
    """
    text = README.read_text(encoding="utf-8")
    m = re.search(r"^##+\s*Forensic Statistics Reference\s*$", text, re.M)
    assert m, "README is missing the 'Forensic Statistics Reference' heading"
    section = text[m.end():]
    nxt = re.search(r"^##\s", section, re.M)
    if nxt:
        section = section[: nxt.start()]
    names = []
    for line in section.splitlines():
        row = re.match(r"\s*\|\s*`([a-z_][a-z0-9_]*)\(\)`", line)
        if row:
            names.append(row.group(1))
    return names


_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _markdown_files() -> list[Path]:
    files = [README, REPO / "CLAUDE.md", REPO / "ARCHITECTURE.md"]
    files += sorted((REPO / "docs").rglob("*.md"))
    return [f for f in files if f.exists()]


def _relative_links(path: Path) -> list[str]:
    out = []
    for target in _LINK_RE.findall(path.read_text(encoding="utf-8")):
        target = target.strip()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = target.split("#", 1)[0].strip()  # drop any anchor
        if target:
            out.append(target)
    return out


# ─────────────────────────────────────────────────────────────────────────
# (a) dispatcher subcommands are documented in the README
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("module", ["paperscope", "paperscope.systematic_review"])
def test_every_subcommand_is_in_readme(module):
    readme = README.read_text(encoding="utf-8")
    subs = _dispatcher_subcommands(module)
    assert subs, f"no subcommands discovered for {module}"
    missing = [s for s in subs if s not in readme]
    assert not missing, f"{module} subcommands absent from README.md: {missing}"


# ─────────────────────────────────────────────────────────────────────────
# (b) forensic reference table names exist in the module
# ─────────────────────────────────────────────────────────────────────────
def test_forensic_reference_functions_exist():
    import paperscope.analysis.forensic_stats as fs

    names = _forensic_reference_functions()
    assert names, "no functions parsed from the forensic reference table"
    missing = [n for n in names if not callable(getattr(fs, n, None))]
    assert not missing, f"README forensic table names missing from module: {missing}"


# ─────────────────────────────────────────────────────────────────────────
# (c) relative markdown links resolve
# ─────────────────────────────────────────────────────────────────────────
def test_relative_markdown_links_resolve():
    broken = []
    for md in _markdown_files():
        for target in _relative_links(md):
            resolved = (md.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{md.relative_to(REPO)} -> {target}")
    assert not broken, "broken relative markdown links:\n" + "\n".join(broken)


# ─────────────────────────────────────────────────────────────────────────
# (d) README badge floor matches pyproject requires-python
# ─────────────────────────────────────────────────────────────────────────
def test_python_floor_matches_pyproject():
    readme = README.read_text(encoding="utf-8")
    badge = re.search(r"python-(\d+\.\d+)\+-blue", readme)
    assert badge, "README is missing the python-X.Y+ badge"

    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    req = re.search(r'requires-python\s*=\s*"[^0-9]*(\d+\.\d+)"', pyproject)
    assert req, "pyproject.toml is missing a parseable requires-python"

    assert badge.group(1) == req.group(1), (
        f"README badge floor {badge.group(1)} != "
        f"pyproject requires-python {req.group(1)}"
    )
