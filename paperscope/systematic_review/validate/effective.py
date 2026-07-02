"""Decisions-of-record: AI JSONL + validation-overrides.jsonl → effective rows.

`validate reconcile` is append-only — it never mutates screening.jsonl /
extraction.jsonl. This module is the single place where the human overrides
are folded back in, so every downstream consumer (PRISMA flow, synthesis
aggregation, the review site, a requeued re-screen) sees the same
decisions-of-record: **human override wins; the AI original is preserved in
provenance fields**, never erased.

Resolution rules:
- overrides are consumed per **stage**: reconcile tags each row with
  `stage: "screening" | "extraction"` and each loader consumes only its own
  lane. Untagged legacy rows are inferred from content — a row carrying any
  corrected_fields input (even a malformed one kept as
  `corrected_fields_raw`) targets extraction and is NEVER read as a
  screening flip;
- the overrides log is append-only, so the LAST row per (record, stage)
  wins whatever its `human` value: a later 'agree' row retracts an earlier
  flip. A row is applied only when it is a flip — explicitly
  (human='flip') or implicitly (it carries a non-empty corrected value);
- an override with an explicit `corrected_decision` sets the decision;
- a bare flip of a binary decision inverts the row's ORIGINAL AI decision
  (`ai_decision` if the row already carries human provenance, else
  `decision`), so re-applying the same override to an already-corrected
  row — e.g. after the documented requeue flow wrote the human decision
  back into screening.jsonl — is IDEMPOTENT, never a re-inversion;
- a bare flip of anything else (e.g. "maybe") is ambiguous — the AI decision
  is kept but marked `human_flip_unresolved: True` rather than silently
  guessed;
- extraction overrides merge `corrected_fields` over the row (identity keys
  are protected), stashing the replaced values in `original_fields`.

Rows without an override pass through unchanged, so a corpus with no
overrides file loads byte-identical to the raw JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

from paperscope.systematic_review.records import load_jsonl, record_id

DEFAULT_OVERRIDES_NAME = "validation-overrides.jsonl"

# Keys corrected_fields may never rewrite — a human override that changed a
# record's identity would silently re-key the whole pipeline.
_IDENTITY_KEYS = {"record_id", "pmid", "id"}

_BINARY_FLIP = {"include": "exclude", "exclude": "include"}


def _corrected_fields(override: dict) -> dict:
    """The override's corrected_fields as a dict (tolerates a JSON string)."""
    cf = override.get("corrected_fields")
    if isinstance(cf, str):
        try:
            cf = json.loads(cf)
        except ValueError:
            return {}
    return cf if isinstance(cf, dict) else {}


def _targets_extraction(override: dict) -> bool:
    """Does this override carry any corrected_fields input?

    True for a parsed dict, for a malformed string kept as
    `corrected_fields_raw`, and for any non-empty raw string value: the
    human typed into the extraction control, so the row targets extraction
    — it must never fall through to a binary screening flip.
    """
    if override.get("corrected_fields_raw"):
        return True
    cf = override.get("corrected_fields")
    if cf is None:
        return False
    if isinstance(cf, dict):
        return bool(cf)
    return bool(str(cf).strip())


def _stage_of(override: dict) -> str:
    """The decision lane an override row targets.

    Prefers the explicit `stage` tag reconcile writes; untagged legacy rows
    are inferred from content (any corrected_fields input -> extraction,
    else screening — the pre-tag behaviour, minus the malformed-fields
    fall-through)."""
    stage = override.get("stage")
    if stage in ("screening", "extraction"):
        return stage
    return "extraction" if _targets_extraction(override) else "screening"


def _is_flip(override: dict) -> bool:
    """Explicit flip, or an implicit one (a non-empty corrected value)."""
    if override.get("human") == "flip":
        return True
    if (override.get("corrected_decision") or "").strip():
        return True
    return _targets_extraction(override)


def _load_overrides(corpus_dir: str | Path | None,
                    overrides_path: str | Path | None,
                    stage: str) -> dict[str, dict]:
    """record_id -> the LAST override row for `stage`, kept only if that
    last row is a flip (an appended 'agree' row retracts an earlier flip).
    Missing file -> {}."""
    if overrides_path is None:
        if corpus_dir is None:
            return {}
        overrides_path = Path(corpus_dir) / DEFAULT_OVERRIDES_NAME
    overrides_path = Path(overrides_path)
    if not overrides_path.exists():
        return {}
    last: dict[str, dict] = {}
    for o in load_jsonl(overrides_path):
        if _stage_of(o) != stage:
            continue
        rid = str(o.get("record_id") or "")
        if rid:
            last[rid] = o  # last row per (record, stage) wins
    return {rid: o for rid, o in last.items() if _is_flip(o)}


def _load_flips(corpus_dir: str | Path | None,
                overrides_path: str | Path | None,
                stage: str = "screening") -> dict[str, dict]:
    """Back-compat alias for _load_overrides (screening lane by default)."""
    return _load_overrides(corpus_dir, overrides_path, stage)


def _apply_screening_flip(row: dict, override: dict) -> dict:
    # The ORIGINAL AI decision: if the row already carries human provenance
    # (e.g. a requeue pass wrote the corrected decision back into
    # screening.jsonl), `ai_decision` holds the AI original — inferring the
    # bare flip against it makes re-application idempotent instead of
    # inverting the human decision back.
    ai_decision = row.get("ai_decision") or row.get("decision", "")
    corrected = override.get("corrected_decision")
    if not corrected:
        if _targets_extraction(override):
            # Extraction-targeted override (including a malformed
            # corrected_fields kept as raw): never a screening flip.
            return row
        corrected = _BINARY_FLIP.get(ai_decision)
    out = dict(row)
    if not corrected:
        out["human_flip_unresolved"] = True
        return out
    out["decision"] = corrected
    out["decided_by"] = "human"
    out["ai_decision"] = ai_decision
    if override.get("note"):
        out["human_note"] = override["note"]
    return out


def load_effective_screening(
    corpus_dir: str | Path,
    *,
    screening_path: str | Path | None = None,
    overrides_path: str | Path | None = None,
) -> list[dict]:
    """Screening decisions-of-record: <corpus>/screening.jsonl with human
    flips from <corpus>/validation-overrides.jsonl applied on top."""
    corpus_dir = Path(corpus_dir)
    rows = load_jsonl(screening_path or corpus_dir / "screening.jsonl")
    flips = _load_overrides(corpus_dir, overrides_path, "screening")
    if not flips:
        return rows
    return [
        _apply_screening_flip(r, flips[record_id(r)]) if record_id(r) in flips else r
        for r in rows
    ]


def load_effective_extraction(
    corpus_dir: str | Path,
    *,
    extraction_path: str | Path | None = None,
    overrides_path: str | Path | None = None,
) -> list[dict]:
    """Extraction rows-of-record: <corpus>/extraction.jsonl with human
    corrected_fields from <corpus>/validation-overrides.jsonl merged in."""
    corpus_dir = Path(corpus_dir)
    rows = load_jsonl(extraction_path or corpus_dir / "extraction.jsonl")
    flips = _load_overrides(corpus_dir, overrides_path, "extraction")
    if not flips:
        return rows
    out: list[dict] = []
    for r in rows:
        override = flips.get(record_id(r))
        if override is None:
            out.append(r)
            continue
        corrected = {
            k: v for k, v in _corrected_fields(override).items() if k not in _IDENTITY_KEYS
        }
        merged = dict(r)
        if not corrected:
            # An extraction-targeted flip whose fields are unusable
            # (malformed JSON, or only identity keys): say so, don't guess.
            merged["human_flip_unresolved"] = True
            out.append(merged)
            continue
        merged["original_fields"] = {k: r.get(k) for k in corrected}
        merged.update(corrected)
        merged["corrected_by"] = "human"
        if override.get("note"):
            merged["human_note"] = override["note"]
        out.append(merged)
    return out


def human_corrected_screening(
    corpus_dir: str | Path | None = None,
    *,
    overrides_path: str | Path | None = None,
) -> dict[str, dict]:
    """record_id -> the human decision-of-record, for resolved flips only.

    Built from the overrides alone (each carries `original_decision`), so a
    requeue/re-screen pass can protect human-adjudicated records even when
    screening.jsonl has since been rewritten. Feed the result to
    `screen.ai_screen.screen_corpus(..., human_corrected=...)`.
    """
    out: dict[str, dict] = {}
    for rid, override in _load_overrides(corpus_dir, overrides_path, "screening").items():
        base = dict(override.get("original_decision") or {})
        effective = _apply_screening_flip(base, override)
        if effective.get("human_flip_unresolved"):
            continue  # nothing concrete to carry forward
        if effective == base:
            continue  # extraction-targeted override: no screening decision
        effective.setdefault("record_id", rid)
        out[rid] = effective
    return out
