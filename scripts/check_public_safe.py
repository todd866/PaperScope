#!/usr/bin/env python3
"""Block private/confidential content from entering this public repo.

Runs over the git-tracked files (what would be published) and fails if any carry
paywall-service references, unpublished-review statistics, private home paths,
real-author forensic naming, private disease-domain terms, or the private
constellation map. Wired as a git pre-commit hook (scripts/install_hooks.sh)
and runnable in CI.

Modes:
  default    scan the tracked files at the tip (plus PDF text where a text
             extractor is available; other binaries are listed, not silently
             skipped)
  --history  additionally scan every blob reachable from HEAD and every commit
             message, so a tip-swap cannot hide a leak below the surface
  --root     scan a different repo (used by the tests against fixture repos)
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/check_public_safe.py"

# (label, regex) — case-insensitive. These must never appear in a public commit.
FORBIDDEN = [
    ("paywall service", r"annas[- ]?archive|sci[-_]?hub|scidb|libgen"),
    ("unpublished review stats", r"\b13,?058\b|\b6,?721\b|\b2,?210\b|\b1,?464\b|67\.5%|96\.7%"),
    ("private home path", r"/Users/[A-Za-z0-9._-]+"),
    ("private project path", r"Desktop/medicine|md-project"),
    ("real forensic author",
     r"rajizadeh|haghighian|fallah|azhar|daneshzad|\byari\b|yazdanpanah|demo_magnesium"),
    ("private constellation map", r"PROJECT-MAP|GLOBAL_PROJECT_MAP|highdimensional"),
    # Disease domain of the private review. The acronyms are matched
    # case-sensitively ((?-i:...)) so ordinary words containing 'als'/'mnd'
    # can't false-positive; the spelled-out forms stay case-insensitive.
    ("disease term", r"(?-i:\bMND\b|\bALS\b)|amyotrophic|motor[\s-]neurone?s?\b"),
]
COMPILED = [(label, re.compile(pat, re.I)) for label, pat in FORBIDDEN]
TEXT_EXT = re.compile(r"\.(py|md|txt|json|ts|tsx|js|mjs|yml|yaml|toml|cfg|sh|tex|bib|sql)$", re.I)
TEXT_BASENAMES = {".gitignore", ".gitattributes", "LICENSE", "Makefile"}


def _is_text(rel: str) -> bool:
    return bool(TEXT_EXT.search(rel)) or rel.rsplit("/", 1)[-1] in TEXT_BASENAMES


def tracked_files(root: Path) -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True,
                             text=True, check=True).stdout
        return [f for f in out.splitlines() if f]
    except Exception:
        return [str(p.relative_to(root)) for p in root.rglob("*")
                if p.is_file() and ".git/" not in str(p) and "__pycache__" not in str(p)]


def _pdf_text(data: bytes) -> "str | None":
    """Extract PDF text via PyMuPDF, else pdftotext; None if neither works."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None
    if fitz is not None:
        try:
            with fitz.open(stream=data, filetype="pdf") as doc:
                return "".join(page.get_text() for page in doc)
        except Exception:
            return None
    if shutil.which("pdftotext"):
        try:
            proc = subprocess.run(["pdftotext", "-", "-"], input=data,
                                  capture_output=True, check=True)
            return proc.stdout.decode(errors="ignore")
        except Exception:
            return None
    return None


def _scan(where: str, text: str, problems: list) -> None:
    for label, rx in COMPILED:
        m = rx.search(text)
        if m:
            problems.append(f"{where}: {label} -> {m.group(0)!r}")


def scan_tip(root: Path) -> "tuple[list, list, int]":
    problems: list = []
    unscanned: list = []
    files = tracked_files(root)
    for rel in files:
        if rel == SELF:
            continue
        path = root / rel
        if _is_text(rel):
            try:
                _scan(rel, path.read_text(errors="ignore"), problems)
            except OSError:
                continue
        elif rel.lower().endswith(".pdf"):
            try:
                text = _pdf_text(path.read_bytes())
            except OSError:
                continue
            if text is None:
                unscanned.append(rel)
            else:
                _scan(rel, text, problems)
        else:
            unscanned.append(rel)
    return problems, unscanned, len(files)


def _history_blobs(root: Path) -> "list[tuple[str, str, bytes]]":
    """(sha, path, data) for every blob reachable from HEAD, deduped by sha."""
    out = subprocess.run(["git", "rev-list", "--objects", "HEAD"], cwd=root,
                         capture_output=True, text=True, check=True).stdout
    paths: dict = {}
    for line in out.splitlines():
        sha, _, path = line.partition(" ")
        if path and sha not in paths:
            paths[sha] = path
    if not paths:
        return []
    batch = subprocess.run(["git", "cat-file", "--batch"], cwd=root,
                           input="\n".join(paths).encode() + b"\n",
                           capture_output=True, check=True).stdout
    blobs = []
    pos = 0
    while pos < len(batch):
        nl = batch.index(b"\n", pos)
        header = batch[pos:nl].decode(errors="ignore").split()
        if len(header) == 3:
            sha, otype, size = header[0], header[1], int(header[2])
            if otype == "blob":
                blobs.append((sha, paths[sha], batch[nl + 1:nl + 1 + size]))
            pos = nl + 1 + size + 1  # content + trailing newline
        else:  # "<sha> missing" — shouldn't happen for rev-list output
            pos = nl + 1
    return blobs


def _commit_messages(root: Path) -> "list[tuple[str, str]]":
    out = subprocess.run(["git", "log", "--format=%H%n%B%x00", "HEAD"], cwd=root,
                         capture_output=True, text=True, check=True).stdout
    messages = []
    for chunk in out.split("\x00"):
        chunk = chunk.strip()
        if chunk:
            sha, _, body = chunk.partition("\n")
            messages.append((sha, body))
    return messages


def scan_history(root: Path) -> "tuple[list, list, int, int]":
    problems: list = []
    unscanned: list = []
    blobs = _history_blobs(root)
    for sha, path, data in blobs:
        if path == SELF:
            continue
        where = f"{sha[:12]}:{path}"
        if _is_text(path):
            _scan(where, data.decode(errors="ignore"), problems)
        elif path.lower().endswith(".pdf"):
            text = _pdf_text(data)
            if text is None:
                unscanned.append(where)
            else:
                _scan(where, text, problems)
        else:
            unscanned.append(where)
    messages = _commit_messages(root)
    for sha, body in messages:
        _scan(f"commit {sha[:12]} message", body, problems)
    return problems, unscanned, len(blobs), len(messages)


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="repo to scan (default: this repo)")
    ap.add_argument("--history", action="store_true",
                    help="also scan all blobs reachable from HEAD and all commit messages")
    args = ap.parse_args(argv)
    root = args.root.resolve()

    problems, unscanned, n_files = scan_tip(root)
    scanned = f"{n_files} tracked files"
    if args.history:
        try:
            hp, hu, n_blobs, n_msgs = scan_history(root)
        except subprocess.CalledProcessError as exc:
            print(f"check_public_safe: --history needs git history: {exc}", file=sys.stderr)
            return 1
        problems.extend(hp)
        unscanned.extend(hu)
        scanned += f", {n_blobs} historical blobs, {n_msgs} commit messages"

    if unscanned:
        print("check_public_safe: NOT scanned (no text extractor / binary format):",
              file=sys.stderr)
        for u in unscanned:
            print("  ~ " + u, file=sys.stderr)
    if problems:
        print("check_public_safe FAILED — refusing (do not commit/publish):", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        return 1
    print(f"check_public_safe: ok ({scanned})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
