"""Field-level disagreement between two raters (roadmap item 3).

Two AI (or human) rater families screen/chart the *same* records; this module
tells you where they agree and, more usefully for adjudication, exactly where
and how they disagree. It is deliberately narrower than the "structured
disagreement surface" sketched in `docs/corpus-knowledge-base.md` §3 — it does
the field-level core (agreement counts, percent agreement, Cohen's kappa for a
categorical decision, and a per-record disagreement listing) that a reviewer
needs to open two rater passes and reconcile them.

Inputs are two lists of JSONL rows (screening.jsonl or extraction.jsonl),
keyed by `record_id` (falls back to pmid/id, via `records.record_id`).

## What it handles

- **Missing records**: a record present for only one rater is reported under
  ``coverage.only_in_a`` / ``only_in_b`` and excluded from field agreement
  (you can't disagree with a rater who never saw the record).
- **List-valued fields** (e.g. ``themes``, ``onset_features``): compared as
  **sets**, order-independent. Auto-detected when either rater's value is a
  list, or forced via ``list_fields``.
- **Normalization** (default on, ``normalize=True``): string values are
  compared case-insensitively with surrounding/internal whitespace collapsed
  (``"  Case  Report " == "case report"``); list elements are normalized the
  same way before the set comparison. Turn it off for byte-exact comparison.

## Cohen's kappa

For one categorical field (``kappa_field``, e.g. ``"decision"``) we report
Cohen's unweighted kappa over the records both raters scored:

    kappa = (p_o - p_e) / (1 - p_e)

where ``p_o`` is observed agreement and ``p_e`` is agreement expected by
chance from each rater's marginal category frequencies. Source: Cohen, J.
(1960). "A coefficient of agreement for nominal scales." *Educational and
Psychological Measurement*, 20(1), 37-46. When both raters used a single
category (so ``1 - p_e == 0``) kappa is undefined and reported as ``None``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from paperscope.systematic_review.records import record_id

COHEN_KAPPA_SOURCE = (
    "Cohen, J. (1960). A coefficient of agreement for nominal scales. "
    "Educational and Psychological Measurement, 20(1), 37-46. "
    "kappa = (p_o - p_e) / (1 - p_e)."
)

NORMALIZATION_NOTE = (
    "Scalars compared case-insensitively with whitespace collapsed and "
    "stripped; list-valued fields compared as sets (order-independent) with "
    "each element normalized the same way. Set normalize=False for byte-exact "
    "comparison."
)

# Identity keys are never compared as data fields.
_IDENTITY_KEYS = {"record_id", "pmid", "id"}


def _norm_scalar(value: Any, normalize: bool) -> str:
    """Canonical string form of a scalar for comparison."""
    s = "" if value is None else str(value)
    if normalize:
        s = re.sub(r"\s+", " ", s.strip().lower())
    else:
        s = s.strip()
    return s


def _norm_list(value: Iterable[Any], normalize: bool) -> frozenset[str]:
    """Canonical set form of a list-valued field."""
    return frozenset(
        n for n in (_norm_scalar(v, normalize) for v in value) if n != ""
    )


def _is_empty(value: Any) -> bool:
    """A field is 'not present for comparison' if it is None, an empty/blank
    string, or an empty list — i.e. the rater charted nothing for it."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    return False


def _values_agree(a: Any, b: Any, *, as_list: bool, normalize: bool) -> bool:
    if as_list:
        av = a if isinstance(a, (list, tuple, set)) else [a]
        bv = b if isinstance(b, (list, tuple, set)) else [b]
        return _norm_list(av, normalize) == _norm_list(bv, normalize)
    return _norm_scalar(a, normalize) == _norm_scalar(b, normalize)


def _index_by_record_id(rows: Iterable[dict]) -> dict[str, dict]:
    """record_id -> row (last wins for duplicate ids)."""
    out: dict[str, dict] = {}
    for r in rows:
        rid = record_id(r)
        if rid:
            out[rid] = r
    return out


def cohen_kappa(
    pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    """Cohen's unweighted kappa for a list of (rater_a, rater_b) category pairs.

    Returns ``{n, observed_agreement, expected_agreement, kappa, categories,
    source}``. ``kappa`` is ``None`` when there are no pairs or when expected
    agreement is 1.0 (a single category used by both raters).
    """
    n = len(pairs)
    if n == 0:
        return {
            "n": 0,
            "observed_agreement": None,
            "expected_agreement": None,
            "kappa": None,
            "categories": [],
            "source": COHEN_KAPPA_SOURCE,
        }
    a_counts: Counter[str] = Counter(a for a, _ in pairs)
    b_counts: Counter[str] = Counter(b for _, b in pairs)
    categories = sorted(set(a_counts) | set(b_counts))
    po = sum(1 for a, b in pairs if a == b) / n
    pe = sum((a_counts[c] / n) * (b_counts[c] / n) for c in categories)
    kappa = None if (1 - pe) == 0 else (po - pe) / (1 - pe)
    return {
        "n": n,
        "observed_agreement": po,
        "expected_agreement": pe,
        "kappa": kappa,
        "categories": categories,
        "source": COHEN_KAPPA_SOURCE,
    }


def compare_raters(
    rows_a: list[dict],
    rows_b: list[dict],
    *,
    fields: list[str] | None = None,
    kappa_field: str | None = None,
    list_fields: Iterable[str] | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    """Compare two raters' rows field-by-field over their shared records.

    Args:
        rows_a, rows_b: JSONL rows from rater A and rater B (screening or
            extraction), each keyed by record_id/pmid/id.
        fields: fields to compare; default = every non-identity key seen in
            either rater's shared-record rows.
        kappa_field: a categorical field to compute Cohen's kappa over
            (e.g. ``"decision"``). None to skip.
        list_fields: fields to force set-comparison on. When None, a field is
            treated as list-valued for a record whenever either rater's value
            is a list.
        normalize: case/whitespace normalization (see NORMALIZATION_NOTE).

    Returns a dict with ``coverage``, per-field ``fields`` stats,
    ``disagreements`` (per-record listing for adjudication), ``kappa``, and a
    ``normalization`` note.
    """
    index_a = _index_by_record_id(rows_a)
    index_b = _index_by_record_id(rows_b)
    ids_a, ids_b = set(index_a), set(index_b)
    in_both = sorted(ids_a & ids_b)

    forced_list = set(list_fields) if list_fields is not None else None

    if fields is None:
        seen: list[str] = []
        seen_set: set[str] = set()
        for rid in in_both:
            for row in (index_a[rid], index_b[rid]):
                for k in row:
                    if k not in _IDENTITY_KEYS and k not in seen_set:
                        seen_set.add(k)
                        seen.append(k)
        fields = seen

    field_stats: dict[str, dict[str, Any]] = {}
    per_record: dict[str, list[dict]] = {rid: [] for rid in in_both}
    kappa_pairs: list[tuple[str, str]] = []

    for field_name in fields:
        n_both = n_agree = n_disagree = n_only_a = n_only_b = n_neither = 0
        field_is_list = False
        for rid in in_both:
            va = index_a[rid].get(field_name)
            vb = index_b[rid].get(field_name)
            a_present = not _is_empty(va)
            b_present = not _is_empty(vb)
            if not a_present and not b_present:
                n_neither += 1
                continue
            if a_present and not b_present:
                n_only_a += 1
                continue
            if b_present and not a_present:
                n_only_b += 1
                continue
            # both present
            as_list = (
                field_name in forced_list
                if forced_list is not None
                else (isinstance(va, list) or isinstance(vb, list))
            )
            field_is_list = field_is_list or as_list
            n_both += 1
            if _values_agree(va, vb, as_list=as_list, normalize=normalize):
                n_agree += 1
            else:
                n_disagree += 1
                per_record[rid].append(
                    {"field": field_name, "rater_a": va, "rater_b": vb}
                )
        field_stats[field_name] = {
            "n_both_present": n_both,
            "n_agree": n_agree,
            "n_disagree": n_disagree,
            "percent_agreement": round(100.0 * n_agree / n_both, 2) if n_both else None,
            "n_only_a": n_only_a,
            "n_only_b": n_only_b,
            "n_neither": n_neither,
            "list_valued": field_is_list,
        }

    kappa_result = None
    if kappa_field is not None:
        for rid in in_both:
            va = index_a[rid].get(kappa_field)
            vb = index_b[rid].get(kappa_field)
            if _is_empty(va) or _is_empty(vb):
                continue
            kappa_pairs.append(
                (_norm_scalar(va, normalize), _norm_scalar(vb, normalize))
            )
        kappa_result = cohen_kappa(kappa_pairs)

    disagreements = [
        {"record_id": rid, "fields": per_record[rid]}
        for rid in in_both
        if per_record[rid]
    ]

    return {
        "coverage": {
            "n_rater_a": len(index_a),
            "n_rater_b": len(index_b),
            "n_in_both": len(in_both),
            "only_in_a": sorted(ids_a - ids_b),
            "only_in_b": sorted(ids_b - ids_a),
        },
        "fields": field_stats,
        "kappa_field": kappa_field,
        "kappa": kappa_result,
        "disagreements": disagreements,
        "normalization": NORMALIZATION_NOTE if normalize else "byte-exact (normalize=False)",
    }


def format_table(result: dict[str, Any]) -> str:
    """Render a compare_raters result as a readable fixed-width text report."""
    cov = result["coverage"]
    lines: list[str] = []
    lines.append(
        f"Coverage: rater A={cov['n_rater_a']}  rater B={cov['n_rater_b']}  "
        f"shared={cov['n_in_both']}  "
        f"only A={len(cov['only_in_a'])}  only B={len(cov['only_in_b'])}"
    )
    lines.append("")
    header = f"{'field':<24} {'both':>5} {'agree':>6} {'disag':>6} {'%agree':>7} {'list':>5}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, s in result["fields"].items():
        pct = "-" if s["percent_agreement"] is None else f"{s['percent_agreement']:.1f}"
        lines.append(
            f"{name:<24} {s['n_both_present']:>5} {s['n_agree']:>6} "
            f"{s['n_disagree']:>6} {pct:>7} {('yes' if s['list_valued'] else ''):>5}"
        )
    kappa = result.get("kappa")
    if kappa is not None:
        kv = kappa["kappa"]
        kstr = "undefined" if kv is None else f"{kv:.3f}"
        lines.append("")
        lines.append(
            f"Cohen's kappa ({result.get('kappa_field')}): {kstr}  "
            f"(n={kappa['n']}, p_o={_fmt(kappa['observed_agreement'])}, "
            f"p_e={_fmt(kappa['expected_agreement'])})"
        )
    n_dis = len(result.get("disagreements", []))
    lines.append("")
    lines.append(f"Records with >=1 field disagreement: {n_dis}")
    return "\n".join(lines)


def _fmt(x: Any) -> str:
    return "-" if x is None else f"{x:.3f}"
