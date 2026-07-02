"""Human adjudication must become the decision-of-record.

Reproduces the HITL dead-end (validation-overrides.jsonl consumed by nothing:
a human flip changed no PRISMA number, no site page, no synthesis table) and
pins the fix:

- `validate.effective.load_effective_screening/-extraction` apply human flips
  on top of the AI JSONL (append-only: source files stay untouched);
- the `prisma`, `aggregate`, and `build-site` CLIs route through them;
- a requeued re-screen carries the human decision forward instead of
  overwriting it — and even a naive overwrite of screening.jsonl cannot make
  the pipeline forget the flip.

  python tests/test_sr_effective_overrides.py    # standalone, prints PASS/FAIL
  pytest tests/test_sr_effective_overrides.py    # under pytest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from paperscope.systematic_review.records import dump_jsonl, load_jsonl  # noqa: E402
from paperscope.systematic_review.screen.ai_screen import screen_corpus  # noqa: E402
from paperscope.systematic_review.ui import build_review_site  # noqa: E402
from paperscope.systematic_review.validate.effective import (  # noqa: E402
    human_corrected_screening,
    load_effective_extraction,
    load_effective_screening,
)
from paperscope.systematic_review.validate.reconcile import reconcile  # noqa: E402
from paperscope.systematic_review.__main__ import main as sr_main  # noqa: E402


RECORDS = [
    {"pmid": "101", "title": "Study one", "abstract": "a", "journal": "J", "year": "2020"},
    {"pmid": "102", "title": "Study two", "abstract": "b", "journal": "J", "year": "2021"},
    {"pmid": "103", "title": "Study three", "abstract": "c", "journal": "J", "year": "2022"},
]
SCREENING = [
    {"pmid": "101", "decision": "include", "themes": ["dx"], "reason": "ai: fits"},
    {"pmid": "102", "decision": "include", "themes": ["dx"], "reason": "ai: fits"},
    {"pmid": "103", "decision": "exclude", "themes": [], "reason": "ai: review article"},
]


def _corpus_with_flip(tmp: Path) -> Path:
    """AI screened 102 as include; the human flipped it to exclude."""
    corpus = tmp / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    dump_jsonl(RECORDS, corpus / "records.jsonl")
    dump_jsonl(SCREENING, corpus / "screening.jsonl")
    export = {
        "102": {
            "reviewed": True,
            "flip": True,
            "corrected_decision": "exclude",
            "note": "wrong population",
        }
    }
    overrides, _requeue = reconcile(SCREENING, export)
    dump_jsonl(overrides, corpus / "validation-overrides.jsonl")
    return corpus


# --- the effective loaders ---------------------------------------------------


def test_effective_screening_applies_human_flip(tmp_path):
    corpus = _corpus_with_flip(tmp_path)
    eff = {d["pmid"]: d for d in load_effective_screening(corpus)}
    assert eff["102"]["decision"] == "exclude"
    assert eff["102"]["decided_by"] == "human"
    assert eff["102"]["ai_decision"] == "include"  # provenance preserved
    assert eff["102"]["human_note"] == "wrong population"
    # untouched rows pass through unchanged
    assert eff["101"] == SCREENING[0]
    assert eff["103"] == SCREENING[2]
    # append-only: the source file is not mutated
    assert load_jsonl(corpus / "screening.jsonl") == SCREENING


def test_effective_screening_without_overrides_is_identity(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    dump_jsonl(SCREENING, corpus / "screening.jsonl")
    assert load_effective_screening(corpus) == SCREENING


def test_flip_without_corrected_decision_inverts_binary(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    dump_jsonl(SCREENING, corpus / "screening.jsonl")
    overrides, _ = reconcile(SCREENING, {"101": {"reviewed": True, "flip": True}})
    dump_jsonl(overrides, corpus / "validation-overrides.jsonl")
    eff = {d["pmid"]: d for d in load_effective_screening(corpus)}
    assert eff["101"]["decision"] == "exclude"  # include <-> exclude
    assert eff["101"]["decided_by"] == "human"


def test_extraction_only_flip_leaves_screening_decision_untouched(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    dump_jsonl(SCREENING, corpus / "screening.jsonl")
    overrides, _ = reconcile(
        SCREENING,
        {"101": {"reviewed": True, "flip": True,
                 "corrected_fields": {"design": "cohort"}}},
    )
    dump_jsonl(overrides, corpus / "validation-overrides.jsonl")
    eff = {d["pmid"]: d for d in load_effective_screening(corpus)}
    # A flip carrying only corrected_fields targets extraction — it must not
    # invert the screening decision (and so must not move the included set).
    assert eff["101"]["decision"] == "include"
    assert "decided_by" not in eff["101"]


def test_flip_of_maybe_without_corrected_decision_is_unresolved(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rows = [{"pmid": "104", "decision": "maybe", "themes": [], "reason": "?"}]
    dump_jsonl(rows, corpus / "screening.jsonl")
    overrides, _ = reconcile(rows, {"104": {"flip": True}})
    dump_jsonl(overrides, corpus / "validation-overrides.jsonl")
    eff = load_effective_screening(corpus)
    # A "maybe" flip with no corrected decision is ambiguous — keep the AI
    # decision but say so, rather than silently guessing.
    assert eff[0]["decision"] == "maybe"
    assert eff[0]["human_flip_unresolved"] is True


def test_effective_extraction_applies_corrected_fields(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rows = [
        {"pmid": "201", "study_design": "cohort", "country": "AU"},
        {"pmid": "202", "study_design": "rct", "country": "US"},
    ]
    dump_jsonl(rows, corpus / "extraction.jsonl")
    export = {
        "202": {
            "reviewed": True,
            "flip": True,
            "corrected_fields": {"study_design": "case report"},
            "note": "misread the design",
        }
    }
    overrides, _ = reconcile(rows, export)
    dump_jsonl(overrides, corpus / "validation-overrides.jsonl")
    eff = {r["pmid"]: r for r in load_effective_extraction(corpus)}
    assert eff["202"]["study_design"] == "case report"
    assert eff["202"]["country"] == "US"  # untouched fields survive
    assert eff["202"]["corrected_by"] == "human"
    assert eff["202"]["original_fields"] == {"study_design": "rct"}
    assert eff["201"] == rows[0]
    assert load_jsonl(corpus / "extraction.jsonl") == rows  # append-only


def test_corrected_fields_cannot_rewrite_identity(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rows = [{"pmid": "201", "study_design": "cohort"}]
    dump_jsonl(rows, corpus / "extraction.jsonl")
    overrides, _ = reconcile(
        rows,
        {"201": {"flip": True, "corrected_fields": {"pmid": "999", "study_design": "rct"}}},
    )
    dump_jsonl(overrides, corpus / "validation-overrides.jsonl")
    eff = load_effective_extraction(corpus)
    assert eff[0]["pmid"] == "201"  # identity keys are protected
    assert eff[0]["study_design"] == "rct"


# --- CLI wiring: prisma / aggregate / build-site -----------------------------


def test_prisma_cli_counts_human_flip(tmp_path):
    corpus = _corpus_with_flip(tmp_path)
    out = tmp_path / "flow.json"
    rc = sr_main(["prisma", "--corpus", str(corpus), "--out", str(out)])
    assert rc == 0
    flow = json.loads(out.read_text())
    # dead-end reproduction: pre-fix this read raw screening.jsonl and
    # reported included=2 / excluded=1, ignoring the human flip.
    assert flow["included_for_charting"] == 1
    assert flow["excluded_at_title_abstract"] == 2


def test_aggregate_cli_uses_corrected_fields(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rows = [
        {"pmid": "201", "study_design": "cohort"},
        {"pmid": "202", "study_design": "rct"},
    ]
    dump_jsonl(rows, corpus / "extraction.jsonl")
    overrides, _ = reconcile(
        rows,
        {"202": {"flip": True, "corrected_fields": {"study_design": "case report"}}},
    )
    dump_jsonl(overrides, corpus / "validation-overrides.jsonl")
    cfg = tmp_path / "review.yaml"
    cfg.write_text(
        "name: mini\n"
        "pcc: {population: p, concept: c, context: x}\n"
        "corpus_dir: ./corpus\n"
        "aggregation:\n"
        "  scalar_counters:\n"
        "    - {field: study_design, name: designs}\n"
    )
    out = tmp_path / "tables.json"
    rc = sr_main(["aggregate", str(cfg), "--corpus", str(corpus), "--out", str(out)])
    assert rc == 0
    tables = json.loads(out.read_text())
    assert tables["designs"].get("case report") == 1
    assert "rct" not in tables["designs"]


def test_build_site_shows_human_decision(tmp_path):
    corpus = _corpus_with_flip(tmp_path)
    out = tmp_path / "site"
    stats = build_review_site(corpus, out, name="Override check")
    # the funnel/count layer sees the decision-of-record
    assert stats["by_decision"] == {"include": 1, "exclude": 2}
    # 102's record page shows the human decision, with provenance
    page = (out / "record" / "102.html").read_text()
    assert ">exclude</span>" in page
    assert "human override" in page and "include" in page
    # 102 sits in the exclude list, not the include list
    assert "102" in (out / "screening" / "exclude.html").read_text()
    assert "102" not in (out / "screening" / "include.html").read_text()


# --- requeue / re-screen protection ------------------------------------------


def test_rescreen_carries_human_decision_forward(tmp_path):
    corpus = _corpus_with_flip(tmp_path)
    export_requeue = reconcile(
        SCREENING,
        {"102": {"reviewed": True, "flip": True, "corrected_decision": "exclude",
                 "note": "wrong population"}},
    )[1]
    assert export_requeue and export_requeue[0]["record_id"] == "102"

    # The AI, re-screening the requeued record, still stubbornly says include.
    def stubborn(record, rubric):
        return {"pmid": record["pmid"], "decision": "include", "themes": ["dx"],
                "reason": "ai: still fits"}

    requeued_records = [r for r in RECORDS if r["pmid"] == "102"]
    protected = human_corrected_screening(corpus)
    out = screen_corpus(requeued_records, None, stubborn, human_corrected=protected)
    assert out[0]["decision"] == "exclude"  # human decision survives the requeue
    assert out[0]["decided_by"] == "human"


def test_effective_screening_survives_naive_rescreen_overwrite(tmp_path):
    corpus = _corpus_with_flip(tmp_path)
    # A careless caller re-screens and overwrites screening.jsonl wholesale,
    # with the AI reverting 102 to include.
    reverted = [dict(d) for d in SCREENING]
    dump_jsonl(reverted, corpus / "screening.jsonl")
    eff = {d["pmid"]: d for d in load_effective_screening(corpus)}
    assert eff["102"]["decision"] == "exclude"  # the flip still wins


if __name__ == "__main__":
    import tempfile

    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)


# --- adversarial-review A4/A5: override lifecycle ----------------------------


def test_requeue_round_trip_keeps_human_decision(tmp_path):
    """AI include -> human bare flip -> requeue re-screen writes the human
    decision into screening.jsonl -> the effective view must still show the
    human decision (the old code re-applied the bare flip on top of the
    already-corrected row and inverted it back)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    dump_jsonl(RECORDS, corpus / "records.jsonl")
    dump_jsonl(SCREENING, corpus / "screening.jsonl")
    overrides, _ = reconcile(SCREENING, {"102": {"reviewed": True, "flip": True}})
    dump_jsonl(overrides, corpus / "validation-overrides.jsonl")

    # sanity: pre-requeue effective view shows the flip
    eff = {d["pmid"]: d for d in load_effective_screening(corpus)}
    assert eff["102"]["decision"] == "exclude"

    # requeue: re-screen everything, human adjudications carried forward,
    # and (as the documented flow allows) screening.jsonl rewritten
    def ai(record, rubric):
        return {"pmid": record["pmid"], "decision": "include", "themes": [],
                "reason": "ai: still fits"}

    protected = human_corrected_screening(corpus)
    rescreened = screen_corpus(RECORDS, None, ai, human_corrected=protected)
    dump_jsonl(rescreened, corpus / "screening.jsonl")

    eff = {d["pmid"]: d for d in load_effective_screening(corpus)}
    assert eff["102"]["decision"] == "exclude"  # NOT inverted back to include
    assert eff["102"]["decided_by"] == "human"
    assert eff["102"]["ai_decision"] == "include"  # provenance intact


def test_malformed_corrected_fields_never_becomes_screening_flip(tmp_path):
    """A flip whose corrected_fields failed to parse targets extraction —
    it must not fall through to a binary screening flip."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    dump_jsonl(SCREENING, corpus / "screening.jsonl")
    overrides, _ = reconcile(
        SCREENING, {"101": {"flip": True, "corrected_fields": "{not json"}}
    )
    dump_jsonl(overrides, corpus / "validation-overrides.jsonl")
    eff = {d["pmid"]: d for d in load_effective_screening(corpus)}
    assert eff["101"]["decision"] == "include"  # NOT inverted
    assert "decided_by" not in eff["101"]


def test_later_agree_clears_earlier_flip(tmp_path):
    """The overrides log is append-only: a human must be able to retract a
    flip by appending an 'agree' row. The LAST row per record wins,
    whatever its value — not the last FLIP."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    dump_jsonl(SCREENING, corpus / "screening.jsonl")
    o1, _ = reconcile(SCREENING, {"102": {"reviewed": True, "flip": True}})
    o2, _ = reconcile(SCREENING, {"102": {"reviewed": True, "flip": False}})
    dump_jsonl(o1 + o2, corpus / "validation-overrides.jsonl")
    eff = {d["pmid"]: d for d in load_effective_screening(corpus)}
    assert eff["102"]["decision"] == "include"  # back to the AI decision
    assert "decided_by" not in eff["102"]


def test_flip_after_agree_still_applies(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    dump_jsonl(SCREENING, corpus / "screening.jsonl")
    o1, _ = reconcile(SCREENING, {"102": {"reviewed": True, "flip": False}})
    o2, _ = reconcile(SCREENING, {"102": {"reviewed": True, "flip": True}})
    dump_jsonl(o1 + o2, corpus / "validation-overrides.jsonl")
    eff = {d["pmid"]: d for d in load_effective_screening(corpus)}
    assert eff["102"]["decision"] == "exclude"


def test_corrected_decision_under_agree_is_implicit_flip(tmp_path):
    """A workbook row exporting a non-empty corrected_decision under
    human='agree' is a correction, not a no-op: reconcile treats it as an
    implicit flip (with a note) instead of silently ignoring it."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    dump_jsonl(SCREENING, corpus / "screening.jsonl")
    overrides, requeue = reconcile(
        SCREENING,
        {"102": {"reviewed": True, "flip": False, "corrected_decision": "exclude"}},
    )
    o = overrides[0]
    assert o["human"] == "flip"
    assert "implicit" in (o.get("note") or "").lower()
    assert requeue and requeue[0]["record_id"] == "102"
    dump_jsonl(overrides, corpus / "validation-overrides.jsonl")
    eff = {d["pmid"]: d for d in load_effective_screening(corpus)}
    assert eff["102"]["decision"] == "exclude"
    assert eff["102"]["decided_by"] == "human"


def test_override_rows_are_stage_tagged(tmp_path):
    screening_overrides, _ = reconcile(SCREENING, {"102": {"flip": True}})
    assert screening_overrides[0].get("stage") == "screening"
    ext_rows = [{"pmid": "201", "study_design": "cohort"}]
    ext_overrides, _ = reconcile(
        ext_rows, {"201": {"flip": True, "corrected_fields": {"study_design": "rct"}}}
    )
    assert ext_overrides[0].get("stage") == "extraction"


def test_screening_stage_flip_does_not_disturb_extraction_rows(tmp_path):
    """A stage-tagged screening flip must not mark same-id extraction rows
    as unresolved — the stages are separate lanes now."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    dump_jsonl(SCREENING, corpus / "screening.jsonl")
    dump_jsonl([{"pmid": "102", "study_design": "cohort"}], corpus / "extraction.jsonl")
    overrides, _ = reconcile(SCREENING, {"102": {"flip": True}})
    dump_jsonl(overrides, corpus / "validation-overrides.jsonl")
    ext = {r["pmid"]: r for r in load_effective_extraction(corpus)}
    assert "human_flip_unresolved" not in ext["102"]
    eff = {d["pmid"]: d for d in load_effective_screening(corpus)}
    assert eff["102"]["decision"] == "exclude"  # the screening flip still lands
