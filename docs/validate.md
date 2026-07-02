# Validate — human-in-the-loop adjudication of AI decisions

The systematic-review pipeline is `search → screen → extract → synthesise`, where `screen` and `extract` are AI-made. `validate` is the missing step between those AI decisions and trusting them: it turns the model's own uncertainty into a human work queue.

## Why this and not an accuracy/kappa layer

A second-rater "accuracy" pass treats validation as a credential ("a human looked at it"). But human screeners rubber-stamp too, so the credential is not evidence. The useful product is different: the AI **self-audits** — surfaces the specific decisions it itself thinks are fragile — and the human adjudicates **only those**, with disagreements reconciled back as work, not notes.

Naming: the *command* is `validate` (the user-facing action); the *method* is an **AI-assisted uncertainty audit plus human adjudication of flagged decisions**, not independent verification. Keep that distinction in any write-up.

## Design constraints

- **SDK-agnostic + deterministic core.** The model self-audit pass runs *outside* the CLI (like `screen`/`extract`); `self_audit.py` defines only the `SelfAuditor` contract. `validate workbook` consumes a precomputed self-audit JSONL.
- **Append-only provenance.** Reconcile never mutates `screening.jsonl`/`extraction.jsonl`; it writes new `validation-overrides.jsonl`, `requeue.jsonl`, `validation-summary.json`.
- **Overrides are consumed, not archived.** `validate.effective` folds the overrides back into *decisions-of-record*; `prisma`, `aggregate`, and `build-site` all read through it. A human flip changes the PRISMA funnel, the synthesis tables, and the review site — the AI original survives only as provenance fields.
- **Generic `record_id`** (falls back to `pmid`/`id`) — not MEDLINE/PMID-bound.
- **Local source first.** Workbook context comes from the local corpus (`records.jsonl` abstracts); `--include-fulltext` backfills open-access abstracts from Europe PMC only. Paywalled full text is never embedded by default.

## Pipeline

```bash
# 1. (your agent SDK) run a SelfAuditor over the decisions -> screening-self-audit.jsonl
#    one {record_id, confidence, flag, reasoning} per decision

# 2. build the scroll-through workbook (AI-flagged decisions sort to the top)
python -m paperscope.systematic_review validate workbook \
  --decisions screening.jsonl --self-audit screening-self-audit.jsonl \
  --rubric validation-rubric.yaml --records records.jsonl \
  --out validation-workbook.html

# 3. (human) rate the friction dimensions; on a flip, set the structured
#    corrected value — a corrected-decision select on screening workbooks, a
#    corrected-fields JSON input on extraction workbooks — plus an optional
#    note. Copy review (JSON) -> export.json

# 4. reconcile (append-only). Write the overrides INTO the corpus dir — that is
#    where the downstream consumers look for them.
python -m paperscope.systematic_review validate reconcile \
  --decisions screening.jsonl --human-export export.json \
  --out <corpus>/validation-overrides.jsonl --requeue <corpus>/requeue.jsonl

# 5. calibration summary (agreement rate, per-dimension tallies)
python -m paperscope.systematic_review validate summary \
  --validation <corpus>/validation-overrides.jsonl --out validation-summary.json

# 6. done — prisma / aggregate / build-site now read the decisions-of-record
#    (AI JSONL + overrides) automatically; no extra step.

# 7. optionally feed requeue.jsonl into a re-screen / re-extract pass. Pass
#    validate.effective.human_corrected_screening(corpus) to
#    screen_corpus(..., human_corrected=...) so the re-screen carries human
#    adjudications forward instead of overwriting them.
```

## From overrides to decisions-of-record

`validate/effective.py` is the single place overrides are folded back in:

- **Stage lanes.** `reconcile` tags every override row with `stage: "screening" | "extraction"` (any corrected-fields input — even one that failed to parse and was kept as `corrected_fields_raw` — targets extraction; otherwise the workbook shape decides). Each effective loader consumes only its own lane, so a `corrected_fields` override, malformed or not, can never invert a screening decision. Untagged legacy rows are inferred from content the same way.
- **Last row wins.** The overrides log is append-only, and the LAST row per (record, stage) wins whatever its `human` value — a later `agree` row retracts an earlier flip. A row is applied only when it is a flip: explicitly (`human: flip`) or implicitly (it carries a non-empty corrected value — `reconcile` records such a row as an implicit flip with a note, rather than silently ignoring a corrected value exported under `agree`).
- `load_effective_screening(corpus_dir)` — `screening.jsonl` + `validation-overrides.jsonl` → screening decisions-of-record. A flip with a `corrected_decision` adopts it; a bare flip of a binary decision inverts the row's **original AI decision** (`ai_decision` when the row already carries human provenance, else `decision`) — so re-applying an override to a row that already holds the human decision (e.g. after a requeue pass wrote it back into `screening.jsonl`) is **idempotent**, never a re-inversion. A bare flip of `maybe` is ambiguous, so the AI decision is kept and marked `human_flip_unresolved: true` rather than silently guessed. Overridden rows carry `decided_by: human`, `ai_decision` (the original), and `human_note`.
- `load_effective_extraction(corpus_dir)` — merges `corrected_fields` over the row (identity keys `record_id`/`pmid`/`id` are protected), stashing replaced values in `original_fields` with `corrected_by: human`. An extraction-targeted flip whose fields are unusable (malformed JSON, or only identity keys) is marked `human_flip_unresolved: true` instead of guessing.
- `human_corrected_screening(corpus_dir)` — record_id → human decision-of-record, built from the overrides alone, for protecting requeued re-screens. `screen_corpus(..., human_corrected=...)` carries these records forward as-is instead of calling the screener, so a requeue pass can never overwrite a human adjudication. Even a wholesale rewrite of `screening.jsonl` cannot make the pipeline forget a flip: the effective loaders reapply the overrides on every read (idempotently — see above).

Rows without an override pass through unchanged; a corpus with no overrides file loads identically to the raw JSONL.

## The friction rubric

A small YAML naming the dimensions a human rates — the generalisation of "rate different friction points" (eligibility, extraction fidelity, claimed-vs-supported task, comparator realism, ...):

```yaml
dimensions:
  - id: eligible
    label: Eligibility correct?
    question: AI/ML diagnostic or surveillance tool, human data, discrimination metric?
    scale: ["yes", "no", "unsure"]
  - id: extraction
    label: Extraction faithful?
    question: Does the recorded operating point match the source?
    scale: ["yes", "no", "unsure"]
```

Quote scale values (`"yes"`/`"no"`) — YAML 1.1 parses bare `yes`/`no` as booleans (the loader coerces them defensively anyway). Omit the rubric and a generic agree/flag pass is used.

## Provenance

The pattern was prototyped on a real review's AI-screening QC: rather than human-rubber-stamping an AI eligibility screen, the decisions were re-examined for self-flagged uncertainty, and disputable headline tags were reported as ranges instead of fixed counts. `validate` is the generalised version.
