"""Corpus knowledge-base bundles (roadmap §6 of docs/corpus-knowledge-base.md).

Turns a review corpus dir into a self-contained, git-safe bundle
(paper-cards.jsonl + clusters.json + manifest.json) that a static site or a
Next.js portal can consume without importing Paperscope at runtime.

Paperscope supplies the mechanics; the caller supplies the domain (which
charted fields land on a card, which field clusters, which fields are quality
flags). Nothing review-specific lives here.
"""

from paperscope.systematic_review.knowledge_base.export import (
    SCHEMA_VERSION,
    build_bundle,
    build_clusters,
    build_manifest,
    build_paper_cards,
    export_knowledge_base,
)

__all__ = [
    "SCHEMA_VERSION",
    "build_bundle",
    "build_clusters",
    "build_manifest",
    "build_paper_cards",
    "export_knowledge_base",
]
