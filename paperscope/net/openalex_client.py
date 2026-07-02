"""One shared OpenAlex client for every paperscope call site.

Before this module, four places (``analysis/author_profile``,
``analysis/related_radar``, ``analysis/journal_targeting``,
``harvest/sources/openalex``) each carried their own copy of the abstract
reconstruction, their own ``OPENALEX_BASE`` + rate limit, and their own GET
wrapper — with *divergent* politeness (1.0s/req vs 0.15s/req) and, worse,
*divergent failure semantics*: some printed and returned ``{}``, one did
``except RequestException: return []`` which makes an API outage
indistinguishable from a genuine "no results".

This module centralises all three:

  * ``get(...)`` — one polite GET (``mailto`` from ``PAPERSCOPE_EMAIL``, one
    documented rate limit) returning a *typed* :class:`OpenAlexResult` so callers
    can tell "API unreachable" (``network_error=True``) apart from "no results"
    (``ok=True`` with an empty ``results``).
  * ``reconstruct_abstract(work)`` — the OpenAlex inverted-index → text helper.

No behaviour change on the success path: a working call returns the same JSON
the old wrappers returned; only the *failure* path is now distinguishable.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


OPENALEX_BASE = "https://api.openalex.org"

# One documented rate limit for all OpenAlex traffic. OpenAlex's polite pool
# permits 10 req/s; 0.15s/req (~6.7 req/s) stays comfortably inside that and
# matches the value most call sites already used. Override via env for tuning.
RATE_LIMIT_DELAY = float(os.environ.get("PAPERSCOPE_OPENALEX_PACE_S", "0.15"))

# Polite-pool contact. Read at call time so a test/env change takes effect.
_DEFAULT_EMAIL = "paperscope@example.com"


def default_email() -> str:
    """Polite-pool mailto: ``PAPERSCOPE_EMAIL`` if set, else the placeholder."""
    return os.environ.get("PAPERSCOPE_EMAIL", _DEFAULT_EMAIL)


@dataclass
class OpenAlexResult:
    """Typed outcome of an OpenAlex request.

    ``ok`` is True only when the API answered with parseable JSON. On any
    transport/HTTP/parse failure ``ok`` is False and ``network_error`` is True,
    so a caller can distinguish an outage from an empty-but-valid result set.
    """

    ok: bool
    data: Dict[str, Any] = field(default_factory=dict)
    network_error: bool = False
    error: str = ""

    @property
    def results(self) -> List[Dict[str, Any]]:
        """The ``results`` list of a list endpoint (``[]`` when absent/failed)."""
        return self.data.get("results", []) or []


def get(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    email: Optional[str] = None,
    timeout: float = 30.0,
    pace_s: Optional[float] = None,
    session: Optional[requests.Session] = None,
) -> OpenAlexResult:
    """Polite GET against OpenAlex, returning a typed :class:`OpenAlexResult`.

    Args:
        endpoint: path under the API base, e.g. ``"works"`` or
            ``"works/doi:10.1/x"``.
        params: query params; ``mailto`` is added if absent.
        email: polite-pool contact; defaults to :func:`default_email`.
        timeout: per-request socket timeout (seconds).
        pace_s: sleep before the request; defaults to :data:`RATE_LIMIT_DELAY`.
            Pass ``0`` to disable pacing (tests).
        session: optional :class:`requests.Session` to reuse a connection pool.
    """
    query = dict(params or {})
    query.setdefault("mailto", email or default_email())
    delay = RATE_LIMIT_DELAY if pace_s is None else pace_s
    if delay:
        time.sleep(delay)
    getter = session.get if session is not None else requests.get
    url = f"{OPENALEX_BASE}/{endpoint.lstrip('/')}"
    try:
        resp = getter(url, params=query, timeout=timeout)
        resp.raise_for_status()
        return OpenAlexResult(ok=True, data=resp.json())
    except (requests.RequestException, ValueError) as exc:
        # RequestException = transport/HTTP; ValueError = non-JSON body. Either
        # way the API gave us nothing usable — a network_error, not "no results".
        return OpenAlexResult(ok=False, network_error=True, error=str(exc))


def reconstruct_abstract(work: Dict[str, Any]) -> str:
    """Reconstruct an abstract from OpenAlex's ``abstract_inverted_index``.

    OpenAlex ships abstracts as ``{word: [positions...]}``; invert that back to
    running text. Returns ``""`` when no inverted index is present.
    """
    inverted = work.get("abstract_inverted_index")
    if not inverted:
        return ""
    word_positions: List[tuple] = []
    for word, positions in inverted.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)
