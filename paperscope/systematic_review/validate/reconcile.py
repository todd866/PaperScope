"""Reconcile a human validation pass back onto the decisions — append-only.

Takes the original decisions (screening.jsonl / extraction.jsonl) and the
human export from the workbook, and produces NEW artefacts. It never mutates
the source decision files: provenance stays append-only, as the rest of the
pipeline does.

Outputs:
- `validation-overrides.jsonl` — one row per human-touched record: the rating
  on each friction dimension, agree/flip, the structured corrected value
  (`corrected_decision` for screening, `corrected_fields` for extraction),
  note, the original decision, and a `stage` tag ("screening"|"extraction")
  saying which decision lane the override targets, so the effective loaders
  never cross lanes (a corrected_fields flip can never invert a screening
  decision). These rows are consumed downstream by `validate.effective`,
  which folds them into the decisions-of-record for PRISMA, aggregation,
  and the review site.

Semantics:
- a non-empty corrected value under human='agree' is an **implicit flip**
  (the human supplied a correction; the checkbox state does not erase it) —
  recorded as human='flip' with a note;
- the log is append-only and the LAST row per (record, stage) wins whatever
  its value, so a later 'agree' row retracts an earlier flip.
- `requeue.jsonl` — the records the human flipped, ready to feed back into a
  re-screen / re-extract pass (closing the loop, rather than a dead note).
- `validation-summary.json` — counts + agreement rate + per-dimension tallies
  (the calibration readout: where does the AI rater disagree with the human?).
"""

from __future__ import annotations

import json

from paperscope.systematic_review.records import record_id


def _parse_corrected_fields(raw) -> dict | None:
    """Corrected fields arrive from the workbook as a JSON string; tolerate a
    dict (programmatic callers). Unparseable input -> None (kept as data by
    the caller under corrected_fields_raw, not silently dropped)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def reconcile(decisions: list[dict], human_export: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Return (overrides, requeue). Append-only: inputs are not modified.

    Each override row is tagged with the `stage` it targets:
    - a row carrying corrected_fields (even unparseable — the human typed
      INTO the extraction control) targets "extraction";
    - otherwise the workbook shape decides: screening-shaped decisions
      (rows with a `decision` key) -> "screening", else "extraction".
    A non-empty corrected value under human='agree' is treated as an
    implicit flip (recorded with a note), not silently dropped.
    """
    by_id = {record_id(d): d for d in decisions}
    workbook_stage = "screening" if any("decision" in d for d in decisions) else "extraction"
    overrides: list[dict] = []
    requeue: list[dict] = []
    for rid, h in human_export.items():
        orig = by_id.get(rid, {})
        flip = bool(h.get("flip"))
        corrected_decision = (h.get("corrected_decision") or "").strip() or None
        raw_fields = h.get("corrected_fields")
        corrected_fields = _parse_corrected_fields(raw_fields)
        has_fields_input = bool(corrected_fields) or (
            isinstance(raw_fields, str) and raw_fields.strip()
        )
        stage = "extraction" if has_fields_input else workbook_stage
        note = h.get("note", "")
        if not flip and (corrected_decision or corrected_fields):
            # The human supplied a corrected value without ticking flip:
            # a correction is a flip in substance — honour it, with a note.
            flip = True
            marker = "implicit flip: corrected value supplied under 'agree'"
            note = f"{note} [{marker}]".strip() if note else marker
        row = {
            "record_id": rid,
            "human": "flip" if flip else "agree",
            "stage": stage,
            "reviewed": bool(h.get("reviewed")),
            "ratings": h.get("ratings", {}),
            "corrected_decision": corrected_decision,
            "corrected_fields": corrected_fields,
            "note": note,
            "original_decision": orig,
        }
        if corrected_fields is None and isinstance(raw_fields, str) \
                and raw_fields.strip():
            row["corrected_fields_raw"] = raw_fields
        overrides.append(row)
        if flip:
            requeue.append({
                "record_id": rid,
                "reason": "human flip on validation",
                "stage": stage,
                "corrected_decision": corrected_decision,
                "note": note,
                "original_decision": orig,
            })
    return overrides, requeue


def summarize(overrides: list[dict]) -> dict:
    """Aggregate overrides into a calibration summary."""
    n = len(overrides)
    reviewed = sum(1 for o in overrides if o.get("reviewed"))
    flipped = sum(1 for o in overrides if o.get("human") == "flip")
    agree = sum(1 for o in overrides if o.get("human") == "agree")
    per_dim: dict[str, dict[str, int]] = {}
    for o in overrides:
        for dim, val in (o.get("ratings") or {}).items():
            per_dim.setdefault(dim, {}).setdefault(str(val), 0)
            per_dim[dim][str(val)] += 1
    return {
        "n_records_touched": n,
        "reviewed": reviewed,
        "flipped": flipped,
        "agreement_rate": (agree / n) if n else None,
        "per_dimension": per_dim,
    }
