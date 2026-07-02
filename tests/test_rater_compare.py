"""Rater comparison: synthetic-fixture TDD.

Covers `paperscope.systematic_review.rater_compare`:

- per-field agreement counts + percent agreement;
- Cohen's kappa against a hand-computed 2x2 (known value 0.40);
- exact per-record disagreement listing (raw values preserved for adjudication);
- missing records in one rater (coverage buckets);
- list-valued fields compared as order-independent sets;
- case/whitespace normalization (documented);
- empty input;
- the CLI subcommand runs end-to-end and emits valid JSON.

All fixtures are synthetic and domain-neutral (this module syncs to the public
repo).

  python tests/test_rater_compare.py     # standalone, prints PASS/FAIL
  pytest tests/test_rater_compare.py     # under pytest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from paperscope.systematic_review.records import dump_jsonl  # noqa: E402
from paperscope.systematic_review.rater_compare import (  # noqa: E402
    cohen_kappa,
    compare_raters,
    format_table,
)
from paperscope.systematic_review.__main__ import main as sr_main  # noqa: E402


# --- Cohen's kappa: hand-computed 2x2 ---------------------------------------
#
#            B=include  B=exclude   row
#  A=include   20          5         25
#  A=exclude   10          15        25
#  col         30          20      N=50
#
#  p_o = (20 + 15) / 50 = 0.70
#  p_e = (25/50)(30/50) + (25/50)(20/50) = 0.30 + 0.20 = 0.50
#  kappa = (0.70 - 0.50) / (1 - 0.50) = 0.20 / 0.50 = 0.40


def _kappa_pairs():
    return (
        [("include", "include")] * 20
        + [("include", "exclude")] * 5
        + [("exclude", "include")] * 10
        + [("exclude", "exclude")] * 15
    )


def test_cohen_kappa_known_value(tmp_path=None):
    result = cohen_kappa(_kappa_pairs())
    assert result["n"] == 50
    assert abs(result["observed_agreement"] - 0.70) < 1e-9
    assert abs(result["expected_agreement"] - 0.50) < 1e-9
    assert abs(result["kappa"] - 0.40) < 1e-9
    assert result["categories"] == ["exclude", "include"]
    assert "Cohen" in result["source"]


def test_cohen_kappa_single_category_is_undefined(tmp_path=None):
    # Both raters used only "include": expected agreement is 1.0, kappa undefined.
    result = cohen_kappa([("include", "include")] * 10)
    assert result["expected_agreement"] == 1.0
    assert result["kappa"] is None


def test_cohen_kappa_empty(tmp_path=None):
    result = cohen_kappa([])
    assert result["n"] == 0 and result["kappa"] is None


def test_compare_raters_kappa_matches_2x2(tmp_path=None):
    # Build rows that realise the exact 2x2 above.
    rows_a, rows_b = [], []
    rid = 0
    for a_dec, b_dec, count in [
        ("include", "include", 20),
        ("include", "exclude", 5),
        ("exclude", "include", 10),
        ("exclude", "exclude", 15),
    ]:
        for _ in range(count):
            rid += 1
            rows_a.append({"pmid": str(rid), "decision": a_dec})
            rows_b.append({"pmid": str(rid), "decision": b_dec})
    result = compare_raters(rows_a, rows_b, kappa_field="decision")
    assert abs(result["kappa"]["kappa"] - 0.40) < 1e-9
    dec = result["fields"]["decision"]
    assert dec["n_both_present"] == 50
    assert dec["n_agree"] == 35
    assert dec["n_disagree"] == 15
    assert dec["percent_agreement"] == 70.0


# --- per-record disagreement listing (exact) --------------------------------


def test_disagreement_listing_exact(tmp_path=None):
    rows_a = [
        {"pmid": "1", "decision": "include", "study_design": "cohort"},
        {"pmid": "2", "decision": "exclude", "study_design": "rct"},
        {"pmid": "3", "decision": "include", "study_design": "case report"},
    ]
    rows_b = [
        {"pmid": "1", "decision": "include", "study_design": "Cohort"},     # agrees (case)
        {"pmid": "2", "decision": "include", "study_design": "rct"},        # decision disagrees
        {"pmid": "3", "decision": "include", "study_design": "case-report"},  # design disagrees (hyphen != space)
    ]
    result = compare_raters(rows_a, rows_b, kappa_field="decision")
    assert result["disagreements"] == [
        {"record_id": "2", "fields": [
            {"field": "decision", "rater_a": "exclude", "rater_b": "include"}]},
        {"record_id": "3", "fields": [
            {"field": "study_design", "rater_a": "case report", "rater_b": "case-report"}]},
    ]
    # study_design agreement: record 1 agrees (Cohort==cohort), record 3 disagrees
    sd = result["fields"]["study_design"]
    assert sd["n_agree"] == 2 and sd["n_disagree"] == 1


# --- missing records ---------------------------------------------------------


def test_missing_records_go_to_coverage(tmp_path=None):
    rows_a = [{"pmid": "1", "decision": "include"},
              {"pmid": "2", "decision": "include"},
              {"pmid": "3", "decision": "exclude"}]
    rows_b = [{"pmid": "2", "decision": "include"},
              {"pmid": "3", "decision": "exclude"},
              {"pmid": "4", "decision": "include"}]
    result = compare_raters(rows_a, rows_b, kappa_field="decision")
    cov = result["coverage"]
    assert cov["n_rater_a"] == 3 and cov["n_rater_b"] == 3
    assert cov["n_in_both"] == 2
    assert cov["only_in_a"] == ["1"]
    assert cov["only_in_b"] == ["4"]
    # kappa computed only over the 2 shared records
    assert result["kappa"]["n"] == 2


# --- list-valued fields ------------------------------------------------------


def test_list_valued_fields_compared_as_sets(tmp_path=None):
    rows_a = [{"pmid": "1", "themes": ["alpha", "beta"]},
              {"pmid": "2", "themes": ["alpha"]}]
    rows_b = [{"pmid": "1", "themes": ["beta", "alpha"]},   # same set, diff order -> agree
              {"pmid": "2", "themes": ["alpha", "gamma"]}]  # different set -> disagree
    result = compare_raters(rows_a, rows_b)
    themes = result["fields"]["themes"]
    assert themes["list_valued"] is True
    assert themes["n_agree"] == 1
    assert themes["n_disagree"] == 1


def test_missingness_tracked_per_field(tmp_path=None):
    rows_a = [{"pmid": "1", "country": "AU"},
              {"pmid": "2", "country": ""},        # blank -> not present
              {"pmid": "3"}]                        # absent
    rows_b = [{"pmid": "1", "country": "au"},       # agrees after normalization
              {"pmid": "2", "country": "US"},       # only B
              {"pmid": "3", "country": "UK"}]       # only B
    result = compare_raters(rows_a, rows_b)
    c = result["fields"]["country"]
    assert c["n_both_present"] == 1
    assert c["n_agree"] == 1
    assert c["n_only_b"] == 2
    assert c["n_only_a"] == 0


# --- normalization toggle ----------------------------------------------------


def test_normalize_off_is_byte_exact(tmp_path=None):
    rows_a = [{"pmid": "1", "study_design": "Cohort"}]
    rows_b = [{"pmid": "1", "study_design": "cohort"}]
    on = compare_raters(rows_a, rows_b)["fields"]["study_design"]
    off = compare_raters(rows_a, rows_b, normalize=False)["fields"]["study_design"]
    assert on["n_agree"] == 1 and on["n_disagree"] == 0
    assert off["n_agree"] == 0 and off["n_disagree"] == 1


# --- empty input -------------------------------------------------------------


def test_empty_inputs(tmp_path=None):
    result = compare_raters([], [], kappa_field="decision")
    assert result["coverage"]["n_in_both"] == 0
    assert result["fields"] == {}
    assert result["disagreements"] == []
    assert result["kappa"]["kappa"] is None
    # table renders without crashing
    assert "Coverage" in format_table(result)


# --- CLI end-to-end ----------------------------------------------------------


def test_cli_rater_compare_end_to_end(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    dump_jsonl([{"pmid": "1", "decision": "include", "themes": ["alpha"]},
                {"pmid": "2", "decision": "exclude", "themes": ["beta"]}], a)
    dump_jsonl([{"pmid": "1", "decision": "include", "themes": ["alpha"]},
                {"pmid": "2", "decision": "include", "themes": ["beta", "gamma"]}], b)
    out = tmp_path / "compare.json"
    rc = sr_main(["rater-compare", "--rater-a", str(a), "--rater-b", str(b),
                  "--kappa-field", "decision", "--out", str(out)])
    assert rc == 0
    result = json.loads(out.read_text())
    assert result["fields"]["decision"]["n_disagree"] == 1
    assert result["kappa_field"] == "decision"
    assert len(result["disagreements"]) == 1


if __name__ == "__main__":
    import inspect
    import tempfile

    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                params = inspect.signature(fn).parameters
                if "tmp_path" in params and fn.__defaults__ is None:
                    with tempfile.TemporaryDirectory() as td:
                        fn(Path(td))
                else:
                    fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
