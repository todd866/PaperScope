"""Journal targeting: rank journals by semantic fit to a paper."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..net import openalex_client
from ..net.openalex_client import reconstruct_abstract as _reconstruct_abstract
from ..text import clean_latex
from ..text.parsing import extract_paragraphs
from ..embed import embed_texts
from ..embed.similarity import cosine_sim


def fetch_journal_abstracts(
    journal_query: str,
    n: int = 100,
    email: str = "paperscope@example.com",
) -> List[str]:
    """Fetch recent abstracts from a journal via OpenAlex.

    Back-compat wrapper around :func:`_fetch_journal_abstracts` that returns only
    the abstract list (drops the network-error flag).

    Args:
        journal_query: Journal name or OpenAlex source ID (e.g. "S12345678").
        n: Number of abstracts to fetch (max 200).
        email: Email for polite pool.

    Returns:
        List of abstract strings.
    """
    return _fetch_journal_abstracts(journal_query, n=n, email=email)[0]


def _fetch_journal_abstracts(
    journal_query: str,
    n: int = 100,
    email: str = "paperscope@example.com",
) -> Tuple[List[str], bool]:
    """Fetch abstracts and report whether OpenAlex was unreachable.

    Returns ``(abstracts, network_error)``. ``network_error`` is True when the
    API call failed (outage), letting the caller tell that apart from a journal
    that genuinely has no indexed abstracts.
    """
    # First resolve journal name to source ID if needed
    if not journal_query.startswith("S"):
        source_id, net_err = _resolve_journal(journal_query, email)
        if not source_id:
            return [], net_err
    else:
        source_id = journal_query

    # Fetch recent works from this source
    resp = openalex_client.get(
        "works",
        {
            "filter": f"primary_location.source.id:{source_id},has_abstract:true",
            "per_page": min(n, 200),
            "sort": "publication_date:desc",
        },
        email=email,
    )
    if resp.network_error:
        return [], True
    abstracts: List[str] = []
    for work in resp.results:
        abstract = _reconstruct_abstract(work)
        if abstract and len(abstract.split()) >= 20:
            abstracts.append(abstract)
    return abstracts, False


def _resolve_journal(name: str, email: str) -> Tuple[Optional[str], bool]:
    """Resolve a journal name to an OpenAlex source ID.

    Returns ``(source_id_or_None, network_error)``.
    """
    resp = openalex_client.get(
        "sources", {"search": name, "per_page": 5}, email=email
    )
    if resp.network_error:
        return None, True
    results = resp.results
    if results:
        return results[0]["id"].split("/")[-1], False
    return None, False


def journal_fit(
    tex_text: str,
    journal_queries: List[str],
    n_per_journal: int = 100,
    email: str = "paperscope@example.com",
    model=None,
) -> Dict:
    """Rank journals by semantic similarity to a paper.

    Embeds paper sections and journal abstracts, computes centroid
    distances.

    Args:
        tex_text: Raw LaTeX source of the paper.
        journal_queries: Journal names or OpenAlex IDs.
        n_per_journal: Abstracts to fetch per journal.
        email: Email for OpenAlex polite pool.
        model: Pre-loaded embedding model.

    Returns:
        Dict with ``rankings`` (sorted by fit) and ``per_section`` breakdown.
    """
    # Extract paper sections
    paras = extract_paragraphs(tex_text)
    if not paras:
        return {"error": "No paragraphs found"}
    paper_texts = [p["text"][:500] for p in paras]
    paper_emb, backend = embed_texts(paper_texts, model=model, show_progress=False)
    paper_centroid = np.mean(paper_emb, axis=0, keepdims=True)

    # Fetch and embed journal abstracts
    rankings: List[Dict] = []
    network_error = False
    for journal in journal_queries:
        print(f"  Fetching abstracts for: {journal}")
        abstracts, net_err = _fetch_journal_abstracts(
            journal, n=n_per_journal, email=email
        )
        if net_err:
            network_error = True
        if not abstracts:
            # Distinguish an outage from a journal that genuinely has no indexed
            # abstracts — a zero fit_score means very different things in each case.
            rankings.append({
                "journal": journal,
                "n_abstracts": 0,
                "fit_score": 0.0,
                "error": "OpenAlex unreachable" if net_err else "No abstracts found",
                "network_error": net_err,
            })
            continue

        journal_emb, _ = embed_texts(abstracts, model=model, show_progress=False)
        journal_centroid = np.mean(journal_emb, axis=0, keepdims=True)

        # Overall fit: centroid similarity
        fit = float(cosine_sim(paper_centroid, journal_centroid)[0, 0])

        # Per-paragraph fit distribution
        para_sims = cosine_sim(paper_emb, journal_centroid)[:, 0]
        rankings.append({
            "journal": journal,
            "n_abstracts": len(abstracts),
            "fit_score": fit,
            "min_paragraph_fit": float(np.min(para_sims)),
            "max_paragraph_fit": float(np.max(para_sims)),
            "std_paragraph_fit": float(np.std(para_sims)),
        })

    rankings.sort(key=lambda x: x.get("fit_score", 0), reverse=True)

    return {
        "rankings": rankings,
        "n_paper_paragraphs": len(paras),
        "backend": backend,
        "network_error": network_error,
    }
