"""Tests for scripts/check_public_safe.py against tiny fixture repos.

Each test builds a throwaway git repo in tmp_path and runs the checker on it
via --root. Forbidden fixture strings are assembled from fragments so this
test file itself stays clean under the tip scan.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_public_safe.py"

# Assembled at runtime so the literals never appear in this file.
DISEASE_WORD = "amyo" + "trophic"
ACRONYM_MND = "M" + "ND"
ACRONYM_ALS = "A" + "LS"
SURNAME = "danesh" + "zad"


def run_check(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True
    )


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def make_repo(tmp_path: Path, files: dict) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    commit_files(repo, files, "initial commit")
    return repo


def commit_files(repo: Path, files: dict, message: str) -> None:
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


def test_clean_repo_passes(tmp_path):
    repo = make_repo(tmp_path, {"notes.md": "All signals look fine; also checked.\n"})
    result = run_check("--root", str(repo))
    assert result.returncode == 0, result.stderr


def test_lowercase_acronym_lookalikes_pass(tmp_path):
    # 'als'/'mnd' inside ordinary words or lowercase prose must not trip the
    # case-sensitive acronym patterns.
    repo = make_repo(
        tmp_path,
        {"doc.md": "signals, morals, commands — als in words is fine\n"},
    )
    result = run_check("--root", str(repo))
    assert result.returncode == 0, result.stderr


def test_tip_disease_word_caught(tmp_path):
    repo = make_repo(tmp_path, {"doc.md": f"a {DISEASE_WORD} case series\n"})
    result = run_check("--root", str(repo))
    assert result.returncode == 1
    assert "disease term" in result.stderr


def test_tip_acronym_caught(tmp_path):
    repo = make_repo(tmp_path, {"doc.md": f"the {ACRONYM_MND} cohort and {ACRONYM_ALS} data\n"})
    result = run_check("--root", str(repo))
    assert result.returncode == 1
    assert "disease term" in result.stderr


def test_tip_surname_caught(tmp_path):
    repo = make_repo(tmp_path, {"doc.md": f"per {SURNAME} et al.\n"})
    result = run_check("--root", str(repo))
    assert result.returncode == 1
    assert "real forensic author" in result.stderr


def test_history_leak_caught_only_by_history_mode(tmp_path):
    # Leak enters history, then is deleted at the tip: the plain scan must
    # pass, --history must fail (a tip-swap can't hide a leaked blob).
    repo = make_repo(tmp_path, {"doc.md": "clean\n"})
    commit_files(repo, {"leak.yaml": f"topic: {DISEASE_WORD}\n"}, "add config")
    (repo / "leak.yaml").unlink()
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "remove config")

    tip = run_check("--root", str(repo))
    assert tip.returncode == 0, tip.stderr

    history = run_check("--root", str(repo), "--history")
    assert history.returncode == 1
    assert "disease term" in history.stderr
    assert "leak.yaml" in history.stderr


def test_history_commit_message_leak_caught(tmp_path):
    repo = make_repo(tmp_path, {"doc.md": "clean\n"})
    commit_files(repo, {"other.md": "still clean\n"}, f"rework the {ACRONYM_MND} pilot")

    tip = run_check("--root", str(repo))
    assert tip.returncode == 0, tip.stderr

    history = run_check("--root", str(repo), "--history")
    assert history.returncode == 1
    assert "commit" in history.stderr and "disease term" in history.stderr


def test_unscanned_binary_is_reported(tmp_path):
    repo = make_repo(tmp_path, {"img.png": b"\x89PNG\r\n\x1a\n0000"})
    result = run_check("--root", str(repo))
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "img.png" in combined, "unscanned binary must be listed, not silently skipped"


def test_pdf_text_is_scanned(tmp_path):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), f"a {DISEASE_WORD} case series")
    pdf_bytes = doc.tobytes()
    doc.close()
    repo = make_repo(tmp_path, {"paper.pdf": pdf_bytes})
    result = run_check("--root", str(repo))
    assert result.returncode == 1
    assert "disease term" in result.stderr
