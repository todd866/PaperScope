"""Knowledge-base exporter: synthetic-fixture TDD.

Covers the roadmap-§6 exporter (`paperscope.systematic_review.knowledge_base`):

- one card per *included* record, with identity / screening / charted / quality
  layers;
- **human overrides are the decisions-of-record** — a validation-overrides flip
  changes card membership and a card's screening decision; a corrected_fields
  override changes a charted value (routed through validate.effective);
- clusters group cards by a configurable charted field, with odd/missing/list
  values handled;
- an empty corpus produces a valid, all-zero bundle;
- the CLI subcommand runs end-to-end and emits valid JSON.

All fixtures are synthetic and domain-neutral — this module syncs to the
public repo, so nothing disease-specific or from any real corpus appears here.

  python tests/test_knowledge_base.py     # standalone, prints PASS/FAIL
  pytest tests/test_knowledge_base.py     # under pytest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from paperscope.systematic_review.records import dump_jsonl, load_jsonl  # noqa: E402
from paperscope.systematic_review.validate.reconcile import reconcile  # noqa: E402
from paperscope.systematic_review.knowledge_base import (  # noqa: E402
    SCHEMA_VERSION,
    build_clusters,
    build_paper_cards,
    export_knowledge_base,
)
from paperscope.systematic_review.__main__ import main as sr_main  # noqa: E402


# --- synthetic, domain-neutral fixtures -------------------------------------

RECORDS = [
    {"pmid": "1", "title": "Widget cohort study", "authors": ["A. One"],
     "year": "2020", "doi": "10.1/a", "journal": "Journal of Widgets"},
    {"pmid": "2", "title": "Gadget trial", "authors": ["B. Two"],
     "year": "2021", "doi": "10.1/b", "journal": "Gadget Review"},
    {"pmid": "3", "title": "Sprocket survey", "authors": ["C. Three"],
     "year": "2022", "doi": "10.1/c", "journal": "Sprocket Quarterly"},
]
SCREENING = [
    {"pmid": "1", "decision": "include", "reason": "in scope", "themes": ["alpha"]},
    {"pmid": "2", "decision": "exclude", "reason": "off scope", "themes": []},
    {"pmid": "3", "decision": "include", "reason": "in scope", "themes": ["beta"]},
]
EXTRACTION = [
    {"pmid": "1", "study_design": "cohort", "country": "AU", "topic": "alpha",
     "quality_flags": ["small_n"]},
    {"pmid": "3", "study_design": "survey", "country": "US", "topic": "beta"},
]


def _write_corpus(tmp: Path, *, screening=SCREENING, extraction=EXTRACTION,
                  override_export=None) -> Path:
    corpus = tmp / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    dump_jsonl(RECORDS, corpus / "records.jsonl")
    dump_jsonl(screening, corpus / "screening.jsonl")
    dump_jsonl(extraction, corpus / "extraction.jsonl")
    if override_export is not None:
        # Caller passes the right base rows (screening or extraction) as `_base`.
        overrides, _ = reconcile(override_export["_base"], override_export["_export"])
        dump_jsonl(overrides, corpus / "validation-overrides.jsonl")
    return corpus


# --- pure card building ------------------------------------------------------


def test_cards_only_for_included_records(tmp_path=None):
    cards = build_paper_cards(RECORDS, SCREENING, EXTRACTION)
    ids = [c["record_id"] for c in cards]
    assert ids == ["1", "3"]  # record 2 was excluded -> no card


def test_card_layers(tmp_path=None):
    cards = {c["record_id"]: c for c in build_paper_cards(RECORDS, SCREENING, EXTRACTION,
                                                          source="medline")}
    c1 = cards["1"]
    # identity
    assert c1["identity"]["title"] == "Widget cohort study"
    assert c1["identity"]["doi"] == "10.1/a"
    assert c1["identity"]["source"] == "medline"
    # screening decision-of-record
    assert c1["screening"]["decision"] == "include"
    assert c1["screening"]["reason"] == "in scope"
    # charted fields (quality field pulled out separately, identity keys dropped)
    assert c1["charted"] == {"study_design": "cohort", "country": "AU", "topic": "alpha"}
    assert "pmid" not in c1["charted"]
    assert "quality_flags" not in c1["charted"]
    # quality flags surfaced only when present
    assert c1["quality_flags"] == {"quality_flags": ["small_n"]}
    assert "quality_flags" not in cards["3"]  # record 3 had none


def test_card_fields_restricts_charted(tmp_path=None):
    cards = {c["record_id"]: c for c in build_paper_cards(
        RECORDS, SCREENING, EXTRACTION, card_fields=["study_design"])}
    assert cards["1"]["charted"] == {"study_design": "cohort"}


def test_extraction_only_corpus_includes_all(tmp_path=None):
    # No screening at all -> every record with an id gets a card.
    cards = build_paper_cards(RECORDS, [], EXTRACTION)
    assert [c["record_id"] for c in cards] == ["1", "2", "3"]
    assert cards[0]["screening"] == {}


# --- human override is the decision-of-record --------------------------------


def test_override_flips_excluded_record_into_cards(tmp_path):
    # AI excluded record 2; a human override includes it. It must now get a
    # card whose decision-of-record is the human decision.
    export = {"2": {"reviewed": True, "flip": True, "corrected_decision": "include",
                    "note": "actually in scope"}}
    corpus = _write_corpus(tmp_path, override_export={"_base": SCREENING, "_export": export})
    manifest = export_knowledge_base(corpus, tmp_path / "kb", generated_at="fixed")
    cards = {json.loads(l)["record_id"]: json.loads(l)
             for l in (tmp_path / "kb" / "paper-cards.jsonl").read_text().splitlines()}
    assert "2" in cards  # flipped in
    assert cards["2"]["screening"]["decision"] == "include"
    assert cards["2"]["screening"]["decided_by"] == "human"
    assert cards["2"]["screening"]["ai_decision"] == "exclude"  # provenance kept
    assert manifest["counts"]["with_human_overrides"] == 1
    assert manifest["counts"]["cards"] == 3
    # append-only: source screening.jsonl untouched
    assert load_jsonl(corpus / "screening.jsonl") == SCREENING


def test_override_changes_charted_value_on_card(tmp_path):
    # A corrected_fields override rewrites a charted field; the card reflects it.
    # A single validation-overrides row is consumed by BOTH effective loaders,
    # so keep the record included explicitly (corrected_decision) while
    # correcting a charted field — mirroring a real human export that reviewed
    # an included record and fixed one charted value.
    export = {"1": {"reviewed": True, "flip": True, "corrected_decision": "include",
                    "corrected_fields": {"study_design": "case report"}}}
    corpus = _write_corpus(tmp_path, override_export={"_base": SCREENING, "_export": export})
    export_knowledge_base(corpus, tmp_path / "kb", generated_at="fixed")
    cards = {json.loads(l)["record_id"]: json.loads(l)
             for l in (tmp_path / "kb" / "paper-cards.jsonl").read_text().splitlines()}
    assert cards["1"]["charted"]["study_design"] == "case report"
    assert cards["1"]["charted"]["corrected_by"] == "human"


# --- clusters ----------------------------------------------------------------


def _cards_for_cluster():
    return [
        {"record_id": "1", "charted": {"topic": "alpha"}},
        {"record_id": "2", "charted": {"topic": "beta"}},
        {"record_id": "3", "charted": {"topic": "alpha"}},
        {"record_id": "4", "charted": {}},          # missing topic
        {"record_id": "5", "charted": {"topic": ""}},  # blank topic
    ]


def test_cluster_grouping_with_odd_and_missing_field(tmp_path=None):
    clusters = build_clusters(_cards_for_cluster(), cluster_field="topic")
    assert clusters["cluster_field"] == "topic"
    by_id = {c["cluster_id"]: c for c in clusters["clusters"]}
    assert by_id["alpha"]["record_ids"] == ["1", "3"]
    assert by_id["alpha"]["paper_count"] == 2
    assert by_id["beta"]["record_ids"] == ["2"]
    # missing AND blank both fall to the unclustered bucket
    assert by_id["unclustered"]["record_ids"] == ["4", "5"]
    assert clusters["n_clusters"] == 3


def test_cluster_list_valued_field_multi_membership(tmp_path=None):
    cards = [
        {"record_id": "1", "charted": {"tags": ["alpha", "beta"]}},
        {"record_id": "2", "charted": {"tags": ["beta"]}},
        {"record_id": "3", "charted": {"tags": []}},  # empty list -> unclustered
    ]
    clusters = build_clusters(cards, cluster_field="tags")
    by_id = {c["cluster_id"]: c for c in clusters["clusters"]}
    assert by_id["alpha"]["record_ids"] == ["1"]
    assert by_id["beta"]["record_ids"] == ["1", "2"]  # record 1 in both
    assert by_id["unclustered"]["record_ids"] == ["3"]


def test_cluster_field_none_single_all_bucket(tmp_path=None):
    clusters = build_clusters(_cards_for_cluster(), cluster_field=None)
    assert clusters["n_clusters"] == 1
    assert clusters["clusters"][0]["cluster_id"] == "all"
    assert clusters["clusters"][0]["paper_count"] == 5


# --- manifest + full export --------------------------------------------------


def test_export_writes_valid_bundle_and_manifest(tmp_path):
    corpus = _write_corpus(tmp_path)
    manifest = export_knowledge_base(
        corpus, tmp_path / "kb", cluster_field="topic", source="medline",
        review_name="Synthetic review", generated_at="fixed-ts",
    )
    kb = tmp_path / "kb"
    # all three files present and valid JSON / JSONL
    cards = [json.loads(l) for l in (kb / "paper-cards.jsonl").read_text().splitlines() if l.strip()]
    clusters = json.loads((kb / "clusters.json").read_text())
    man = json.loads((kb / "manifest.json").read_text())
    assert len(cards) == 2
    assert clusters["cluster_field"] == "topic"
    # manifest counts + provenance
    assert man["schema_version"] == SCHEMA_VERSION
    assert man["review_name"] == "Synthetic review"
    assert man["counts"] == {
        "records": 3, "screened": 3, "included": 2, "excluded": 1,
        "cards": 2, "clusters": clusters["n_clusters"], "with_human_overrides": 0,
    }
    assert man["provenance"]["generated_at"] == "fixed-ts"
    assert man["provenance"]["generator"].endswith("knowledge_base")
    assert man["generated_from"]["records"] == "records.jsonl"
    assert man["generated_from"]["screening"] == "screening.jsonl"
    assert man["generated_from"]["extraction"] == "extraction.jsonl"


def test_synthesis_tables_cross_check_surfaced(tmp_path):
    corpus = _write_corpus(tmp_path)
    (corpus / "synthesis-tables.json").write_text(json.dumps({"corpus_n": 2}))
    manifest = export_knowledge_base(corpus, tmp_path / "kb", generated_at="x")
    assert manifest["synthesis_tables"] == {"present": True, "corpus_n": 2}
    assert manifest["generated_from"]["synthesis_tables"] == "synthesis-tables.json"


def test_empty_corpus_produces_valid_zero_bundle(tmp_path):
    corpus = tmp_path / "empty"
    corpus.mkdir()
    manifest = export_knowledge_base(corpus, tmp_path / "kb", cluster_field="topic",
                                     generated_at="x")
    kb = tmp_path / "kb"
    cards_text = (kb / "paper-cards.jsonl").read_text()
    assert cards_text.strip() == ""  # no cards, but the file exists
    clusters = json.loads((kb / "clusters.json").read_text())
    assert clusters["n_clusters"] == 0
    assert manifest["counts"]["cards"] == 0
    assert manifest["counts"]["records"] == 0


# --- CLI end-to-end ----------------------------------------------------------


def test_cli_knowledge_base_end_to_end(tmp_path):
    corpus = _write_corpus(tmp_path)
    out = tmp_path / "kb"
    rc = sr_main(["knowledge-base", "--corpus", str(corpus), "--out", str(out),
                  "--cluster-field", "topic", "--source", "medline"])
    assert rc == 0
    # every emitted file is valid JSON / JSONL
    [json.loads(l) for l in (out / "paper-cards.jsonl").read_text().splitlines() if l.strip()]
    json.loads((out / "clusters.json").read_text())
    man = json.loads((out / "manifest.json").read_text())
    assert man["counts"]["cards"] == 2


if __name__ == "__main__":
    import tempfile

    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                import inspect

                if "tmp_path" in inspect.signature(fn).parameters and \
                        fn.__defaults__ is None:
                    with tempfile.TemporaryDirectory() as td:
                        fn(Path(td))
                else:
                    fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)


def test_manifest_counts_human_override_hidden_by_card_fields(tmp_path):
    """B8: with card_fields restricting the charted layer, a human
    corrected_fields override on a field NOT in card_fields must still be
    counted in with_human_overrides — the count comes from the effective
    rows, not the filtered card body."""
    corpus = _write_corpus(
        tmp_path,
        override_export={
            "_base": EXTRACTION,
            "_export": {"1": {"flip": True,
                              "corrected_fields": {"study_design": "rct"}}},
        },
    )
    out = tmp_path / "kb"
    manifest = export_knowledge_base(corpus, out, card_fields=["country"])
    # the corrected field is invisible on the card...
    cards = [json.loads(l) for l in (out / "paper-cards.jsonl").read_text().splitlines()]
    card1 = next(c for c in cards if c["record_id"] == "1")
    assert "study_design" not in card1["charted"]
    # ...but the human override is still counted
    assert manifest["counts"]["with_human_overrides"] == 1
