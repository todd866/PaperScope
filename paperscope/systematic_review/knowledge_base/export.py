"""Knowledge-base exporter: a review corpus dir → a self-contained KB bundle.

This is roadmap §6 of `docs/corpus-knowledge-base.md`. It turns the JSONL
pipeline artefacts (records / screening / extraction, plus an optional
synthesis-tables.json) into a small, git-safe, self-describing bundle that a
static site or a Next.js portal can consume *without* importing Paperscope at
runtime:

    <out>/
      paper-cards.jsonl   one card per included record
      clusters.json       cards grouped by a configurable charted field
      manifest.json       counts, provenance, generated-from paths, schema ver.

Two layers, deliberately separated for testability:

- **pure functions** (`build_paper_cards`, `build_clusters`, `build_manifest`,
  `build_bundle`) take already-loaded rows and return plain data. No file IO,
  no clocks. These carry the schema intent.
- **`export_knowledge_base`** does the IO: it loads screening/extraction
  through `validate.effective` so **human overrides are the decisions-of-record**
  (a human include↔exclude flip changes card membership; a `corrected_fields`
  override changes a charted value on the card), then writes the three files.

Everything here is generic. The caller supplies the domain: which charted
fields land on a card (`card_fields`), which field defines a cluster
(`cluster_field`), which fields are quality flags (`quality_flag_fields`).
Nothing about any particular review — disease, rubric, cluster names — lives in
this module.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from paperscope.systematic_review.records import load_jsonl, record_id
from paperscope.systematic_review.validate.effective import (
    DEFAULT_OVERRIDES_NAME,
    load_effective_extraction,
    load_effective_screening,
)

SCHEMA_VERSION = "kb-1.0"
GENERATOR = "paperscope.systematic_review.knowledge_base"

# Sensible defaults for a MEDLINE-shaped corpus; every one is overridable.
DEFAULT_IDENTITY_FIELDS = ["title", "authors", "year", "doi", "pmid", "journal"]
DEFAULT_INCLUDE_DECISIONS = ("include",)
DEFAULT_QUALITY_FLAG_FIELDS = ["quality_flags"]
DEFAULT_UNCLUSTERED = "unclustered"

# Keys that identify a record — never charted onto a card body, never treated
# as a comparable/charted field.
_IDENTITY_KEYS = {"record_id", "pmid", "id"}


def _index_by_record_id(rows: Iterable[dict]) -> dict[str, dict]:
    """record_id -> row (last wins for duplicate ids)."""
    out: dict[str, dict] = {}
    for r in rows:
        rid = record_id(r)
        if rid:
            out[rid] = r
    return out


def _is_included(screening_row: dict | None, include_decisions: tuple[str, ...]) -> bool:
    """A record is included iff its decision-of-record is in `include_decisions`.

    Records with no screening row at all are treated as included, so an
    extraction-only corpus (no screening.jsonl) still produces cards.
    """
    if screening_row is None:
        return True
    return str(screening_row.get("decision", "")) in include_decisions


def build_paper_cards(
    records: list[dict],
    screening: list[dict],
    extraction: list[dict],
    *,
    identity_fields: list[str] | None = None,
    card_fields: list[str] | None = None,
    quality_flag_fields: list[str] | None = None,
    include_decisions: Iterable[str] | None = None,
    source: str | None = None,
) -> list[dict]:
    """One card per *included* record.

    A card has four layers, matching the data model in
    `docs/corpus-knowledge-base.md`:

    - ``identity``: record metadata (title/authors/year/doi/pmid/journal, +
      an optional ``source`` database tag);
    - ``screening``: the screening **decision-of-record** row (decision, plus
      any provenance the effective loader added — ``decided_by``,
      ``ai_decision``, ``human_note``, reason, themes);
    - ``charted``: the extracted fields (extraction row minus identity keys and
      minus quality-flag fields), optionally restricted to ``card_fields``;
    - ``quality_flags``: only present when the extraction row carries one of
      ``quality_flag_fields``.

    ``screening`` and ``extraction`` should be the *effective* rows (human
    overrides already folded in) — `export_knowledge_base` guarantees that.
    """
    identity_fields = identity_fields if identity_fields is not None else DEFAULT_IDENTITY_FIELDS
    quality_flag_fields = (
        quality_flag_fields if quality_flag_fields is not None else DEFAULT_QUALITY_FLAG_FIELDS
    )
    include_decisions = tuple(
        include_decisions if include_decisions is not None else DEFAULT_INCLUDE_DECISIONS
    )
    quality_set = set(quality_flag_fields)

    screen_by_id = _index_by_record_id(screening)
    extract_by_id = _index_by_record_id(extraction)

    cards: list[dict] = []
    for rec in records:
        rid = record_id(rec)
        if not rid:
            continue
        screen_row = screen_by_id.get(rid)
        if not _is_included(screen_row, include_decisions):
            continue

        identity = {f: rec.get(f) for f in identity_fields if f in rec}
        if source is not None:
            identity["source"] = source

        screening_block: dict[str, Any] = {}
        if screen_row is not None:
            screening_block = {
                k: v for k, v in screen_row.items() if k not in _IDENTITY_KEYS
            }

        ext_row = extract_by_id.get(rid, {})
        if card_fields is not None:
            charted_keys = [f for f in card_fields if f in ext_row]
        else:
            charted_keys = [
                k for k in ext_row if k not in _IDENTITY_KEYS and k not in quality_set
            ]
        charted = {k: ext_row[k] for k in charted_keys if k not in quality_set}

        quality_flags = {k: ext_row[k] for k in quality_flag_fields if k in ext_row}

        card: dict[str, Any] = {
            "record_id": rid,
            "identity": identity,
            "screening": screening_block,
            "charted": charted,
        }
        if quality_flags:
            card["quality_flags"] = quality_flags
        cards.append(card)
    return cards


def _cluster_keys(value: Any, unclustered_value: str) -> list[str]:
    """The cluster membership key(s) for a charted value.

    - a scalar becomes one key (stringified, stripped); empty → unclustered;
    - a list becomes one key per non-empty element (multi-membership), so a
      card tagged with several clusters/topics lands in each; empty list →
      unclustered;
    - None / missing → unclustered.
    """
    if value is None:
        return [unclustered_value]
    if isinstance(value, (list, tuple)):
        keys = [str(v).strip() for v in value if str(v).strip()]
        return keys or [unclustered_value]
    key = str(value).strip()
    return [key] if key else [unclustered_value]


def build_clusters(
    cards: list[dict],
    *,
    cluster_field: str | None = None,
    unclustered_value: str = DEFAULT_UNCLUSTERED,
) -> dict[str, Any]:
    """Group cards by ``card['charted'][cluster_field]`` — declarative, like the
    aggregate layer.

    Returns ``{"cluster_field", "n_clusters", "clusters": [...]}`` where each
    cluster is ``{"cluster_id", "paper_count", "record_ids"}`` (record_ids and
    clusters both sorted for a deterministic, diff-stable bundle).

    If ``cluster_field`` is None, all cards fall into a single ``"all"``
    cluster, so the bundle always contains a valid clusters.json.
    """
    buckets: dict[str, list[str]] = {}
    for card in cards:
        rid = card.get("record_id", "")
        if cluster_field is None:
            keys = ["all"]
        else:
            keys = _cluster_keys(card.get("charted", {}).get(cluster_field), unclustered_value)
        for key in keys:
            buckets.setdefault(key, []).append(rid)

    clusters = [
        {
            "cluster_id": key,
            "paper_count": len(ids),
            "record_ids": sorted(ids),
        }
        for key, ids in sorted(buckets.items())
    ]
    return {
        "cluster_field": cluster_field,
        "n_clusters": len(clusters),
        "clusters": clusters,
    }


def build_manifest(
    *,
    records: list[dict],
    screening: list[dict],
    cards: list[dict],
    clusters: dict[str, Any],
    extraction: list[dict] | None = None,
    include_decisions: Iterable[str] | None = None,
    review_name: str | None = None,
    generated_from: dict[str, str] | None = None,
    corpus_dir: str | None = None,
    synthesis_tables: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Counts + provenance for the bundle.

    ``with_human_overrides`` counts cards whose decision-of-record or
    extracted fields came from a human (the effective loaders stamp
    ``decided_by`` / ``corrected_by`` = ``"human"``), so the manifest
    surfaces how much of the bundle is human-adjudicated at a glance.
    The count reads the *effective rows* (pass ``extraction``), not the
    card body — a ``card_fields`` restriction that hides the corrected
    field must not hide the human override from the count.
    """
    include_decisions = tuple(
        include_decisions if include_decisions is not None else DEFAULT_INCLUDE_DECISIONS
    )
    screen_by_id = _index_by_record_id(screening)
    n_screened = len(screen_by_id)
    n_included_screen = sum(
        1 for r in screen_by_id.values() if str(r.get("decision", "")) in include_decisions
    )
    n_excluded_screen = n_screened - n_included_screen

    extract_by_id = _index_by_record_id(extraction or [])
    n_human = 0
    for c in cards:
        rid = c.get("record_id", "")
        ext_row = extract_by_id.get(rid, {})
        if (
            c.get("screening", {}).get("decided_by") == "human"
            or ext_row.get("corrected_by") == "human"
            # back-compat: callers that pass no extraction rows still get
            # the (card-visible) charted-layer signal
            or c.get("charted", {}).get("corrected_by") == "human"
        ):
            n_human += 1

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "review_name": review_name,
        "counts": {
            "records": len(records),
            "screened": n_screened,
            "included": n_included_screen,
            "excluded": n_excluded_screen,
            "cards": len(cards),
            "clusters": clusters.get("n_clusters", 0),
            "with_human_overrides": n_human,
        },
        "cluster_field": clusters.get("cluster_field"),
        "generated_from": generated_from or {},
        "provenance": {
            "generator": GENERATOR,
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
            "corpus_dir": corpus_dir,
        },
    }
    if synthesis_tables is not None:
        # Cross-check only: we don't consume the tables for cards, but surfacing
        # the source corpus_n helps a consumer notice a stale export.
        manifest["synthesis_tables"] = {
            "present": True,
            "corpus_n": synthesis_tables.get("corpus_n"),
        }
    return manifest


def build_bundle(
    records: list[dict],
    screening: list[dict],
    extraction: list[dict],
    *,
    cluster_field: str | None = None,
    identity_fields: list[str] | None = None,
    card_fields: list[str] | None = None,
    quality_flag_fields: list[str] | None = None,
    include_decisions: Iterable[str] | None = None,
    unclustered_value: str = DEFAULT_UNCLUSTERED,
    source: str | None = None,
    review_name: str | None = None,
    generated_from: dict[str, str] | None = None,
    corpus_dir: str | None = None,
    synthesis_tables: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Pure assembly of the bundle from already-loaded rows.

    Returns ``{"paper_cards", "clusters", "manifest"}``. No IO — the same
    inputs always produce the same output (except ``generated_at``, which is
    pinnable for reproducible tests).
    """
    cards = build_paper_cards(
        records,
        screening,
        extraction,
        identity_fields=identity_fields,
        card_fields=card_fields,
        quality_flag_fields=quality_flag_fields,
        include_decisions=include_decisions,
        source=source,
    )
    clusters = build_clusters(
        cards, cluster_field=cluster_field, unclustered_value=unclustered_value
    )
    manifest = build_manifest(
        records=records,
        screening=screening,
        cards=cards,
        clusters=clusters,
        extraction=extraction,
        include_decisions=include_decisions,
        review_name=review_name,
        generated_from=generated_from,
        corpus_dir=corpus_dir,
        synthesis_tables=synthesis_tables,
        generated_at=generated_at,
    )
    return {"paper_cards": cards, "clusters": clusters, "manifest": manifest}


def export_knowledge_base(
    corpus_dir: str | Path,
    out_dir: str | Path,
    *,
    cluster_field: str | None = None,
    identity_fields: list[str] | None = None,
    card_fields: list[str] | None = None,
    quality_flag_fields: list[str] | None = None,
    include_decisions: Iterable[str] | None = None,
    unclustered_value: str = DEFAULT_UNCLUSTERED,
    source: str | None = None,
    review_name: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Load a corpus dir, build the bundle, write the three files.

    Screening and extraction are loaded through `validate.effective`, so human
    overrides in ``validation-overrides.jsonl`` are the decisions-of-record:
    a human include↔exclude flip changes which records get cards; a
    ``corrected_fields`` override changes charted values on a card. The raw
    JSONL is never mutated.

    Returns the manifest dict (also written to ``<out>/manifest.json``).
    """
    corpus_dir = Path(corpus_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = corpus_dir / "records.jsonl"
    screening_path = corpus_dir / "screening.jsonl"
    extraction_path = corpus_dir / "extraction.jsonl"
    overrides_path = corpus_dir / DEFAULT_OVERRIDES_NAME
    synthesis_path = corpus_dir / "synthesis-tables.json"

    records = load_jsonl(records_path) if records_path.exists() else []
    screening = load_effective_screening(corpus_dir) if screening_path.exists() else []
    extraction = load_effective_extraction(corpus_dir) if extraction_path.exists() else []
    synthesis_tables = (
        json.loads(synthesis_path.read_text()) if synthesis_path.exists() else None
    )

    generated_from: dict[str, str] = {}
    if records_path.exists():
        generated_from["records"] = records_path.name
    if screening_path.exists():
        generated_from["screening"] = screening_path.name
    if extraction_path.exists():
        generated_from["extraction"] = extraction_path.name
    if overrides_path.exists():
        generated_from["overrides"] = overrides_path.name
    if synthesis_path.exists():
        generated_from["synthesis_tables"] = synthesis_path.name

    bundle = build_bundle(
        records,
        screening,
        extraction,
        cluster_field=cluster_field,
        identity_fields=identity_fields,
        card_fields=card_fields,
        quality_flag_fields=quality_flag_fields,
        include_decisions=include_decisions,
        unclustered_value=unclustered_value,
        source=source,
        review_name=review_name,
        generated_from=generated_from,
        corpus_dir=str(corpus_dir),
        synthesis_tables=synthesis_tables,
        generated_at=generated_at,
    )

    cards_path = out_dir / "paper-cards.jsonl"
    with cards_path.open("w") as f:
        for card in bundle["paper_cards"]:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")
    (out_dir / "clusters.json").write_text(
        json.dumps(bundle["clusters"], indent=1, ensure_ascii=False)
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(bundle["manifest"], indent=1, ensure_ascii=False)
    )
    return bundle["manifest"]
