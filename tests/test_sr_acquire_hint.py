"""Help/reality sync for the acquire → browser-harvest handoff.

The acquire pipeline prints a "run this next" hint for the paywalled tail.
That hint must be a real, parseable CLI invocation — it used to name a
`--fetch-paywalled` flag that never existed, so following it produced an
argparse error. These tests extract the hinted command from acquire's output
and parse it against the actual CLI, and pin the institutional-ToS caution in
the browser-harvest help text.

  pytest tests/test_sr_acquire_hint.py
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from paperscope.systematic_review.acquire.pipeline import acquire  # noqa: E402
from paperscope.systematic_review.__main__ import _cmd_browser_harvest, build_parser  # noqa: E402


def test_paywalled_hint_command_parses(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rec = {"pmid": "101", "doi": "10.1000/x", "title": "Paywalled study"}
    (corpus / "included.jsonl").write_text(json.dumps(rec) + "\n")

    # Offline run: no OA fetch, no text extraction — just the EZProxy queue,
    # which is the phase that prints the hint.  The public build bakes in no
    # institutional proxy, so a placeholder host is passed explicitly.
    report = acquire(
        review_name="hint-check",
        corpus_dir=corpus,
        ezproxy_host="ezproxy.example.edu",
        fetch_oa=False,
        extract_text_pdfs=False,
        verbose=True,
    )
    assert report.queued_for_ezproxy == 1
    out = capsys.readouterr().out
    m = re.search(r"`python -m paperscope\.systematic_review ([^`]+)`", out)
    assert m, f"no runnable hint found in acquire output:\n{out}"

    # The hinted command must parse against the real CLI (argparse SystemExits
    # on an unknown subcommand or flag) and land on browser-harvest.
    args = build_parser().parse_args(shlex.split(m.group(1)))
    assert args.fn is _cmd_browser_harvest


def test_browser_harvest_help_carries_tos_caution():
    proc = subprocess.run(
        [sys.executable, "-m", "paperscope.systematic_review", "browser-harvest", "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    help_text = proc.stdout.lower()
    assert "license" in help_text, "browser-harvest --help must warn about publisher license terms"
    assert "institution" in help_text, "browser-harvest --help must name the institutional risk"


if __name__ == "__main__":
    sys.exit(subprocess.run([sys.executable, "-m", "pytest", __file__, "-q"]).returncode)
