"""Tests for the shared OpenAlex client and its callers' error surfacing.

The client centralises what were four divergent GET wrappers. The key new
guarantee is a *typed* outcome: an API outage (``network_error=True``) is
distinguishable from a genuine empty result set. These tests pin that, and pin
that ``related_radar`` and ``journal_targeting`` surface the distinction.

Network is mocked, so they run anywhere.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import requests

from paperscope.net import openalex_client as oac


def _resp(json_data, *, raise_exc=None):
    r = MagicMock()
    if raise_exc is not None:
        r.raise_for_status.side_effect = raise_exc
    else:
        r.raise_for_status.return_value = None
    r.json.return_value = json_data
    return r


# ---------------------------------------------------------------------------
# get(): typed outcomes
# ---------------------------------------------------------------------------


def test_get_success_returns_ok_with_results(monkeypatch):
    monkeypatch.setattr(
        oac.requests, "get",
        lambda url, params=None, timeout=None: _resp({"results": [{"id": "W1"}]}),
    )
    out = oac.get("works", {"search": "x"}, pace_s=0)
    assert out.ok is True
    assert out.network_error is False
    assert out.results == [{"id": "W1"}]


def test_get_http_error_is_network_error(monkeypatch):
    monkeypatch.setattr(
        oac.requests, "get",
        lambda *a, **k: _resp(None, raise_exc=requests.HTTPError("500 Server Error")),
    )
    out = oac.get("works", pace_s=0)
    assert out.ok is False
    assert out.network_error is True
    assert out.results == []          # empty, but NOT because "no results"
    assert "500" in out.error


def test_get_timeout_is_network_error(monkeypatch):
    def boom(*a, **k):
        raise requests.Timeout("read timed out")

    monkeypatch.setattr(oac.requests, "get", boom)
    out = oac.get("works", pace_s=0)
    assert out.network_error is True
    assert "timed out" in out.error


def test_get_non_json_body_is_network_error(monkeypatch):
    r = MagicMock()
    r.raise_for_status.return_value = None
    r.json.side_effect = ValueError("Expecting value")
    monkeypatch.setattr(oac.requests, "get", lambda *a, **k: r)
    out = oac.get("works", pace_s=0)
    assert out.network_error is True


def test_get_adds_mailto_from_env(monkeypatch):
    monkeypatch.setenv("PAPERSCOPE_EMAIL", "me@here.test")
    captured: dict = {}

    def cap(url, params=None, timeout=None):
        captured.update(params or {})
        return _resp({"results": []})

    monkeypatch.setattr(oac.requests, "get", cap)
    oac.get("works", {"search": "x"}, pace_s=0)
    assert captured["mailto"] == "me@here.test"


def test_explicit_email_param_overrides_env(monkeypatch):
    monkeypatch.setenv("PAPERSCOPE_EMAIL", "env@x.test")
    captured: dict = {}

    def cap(url, params=None, timeout=None):
        captured.update(params or {})
        return _resp({"results": []})

    monkeypatch.setattr(oac.requests, "get", cap)
    oac.get("works", email="explicit@x.test", pace_s=0)
    assert captured["mailto"] == "explicit@x.test"


def test_session_is_used_when_provided(monkeypatch):
    sess = MagicMock()
    sess.get.return_value = _resp({"results": [{"id": "W"}]})
    monkeypatch.setattr(
        oac.requests, "get",
        MagicMock(side_effect=AssertionError("module requests.get must not be used")),
    )
    out = oac.get("works", pace_s=0, session=sess)
    assert out.results == [{"id": "W"}]
    sess.get.assert_called_once()


# ---------------------------------------------------------------------------
# reconstruct_abstract()
# ---------------------------------------------------------------------------


def test_reconstruct_abstract_roundtrip():
    work = {"abstract_inverted_index": {"Hello": [0], "world": [1]}}
    assert oac.reconstruct_abstract(work) == "Hello world"


def test_reconstruct_abstract_absent_index():
    assert oac.reconstruct_abstract({}) == ""
    assert oac.reconstruct_abstract({"abstract_inverted_index": None}) == ""


# ---------------------------------------------------------------------------
# author_profile._oa_get: back-compat adapter contract (citation_uptake relies
# on {} on failure, dict on success)
# ---------------------------------------------------------------------------


def test_author_profile_oa_get_returns_empty_on_error(monkeypatch):
    from paperscope.analysis import author_profile as ap

    monkeypatch.setattr(
        ap.openalex_client, "get",
        lambda *a, **k: oac.OpenAlexResult(ok=False, network_error=True, error="down"),
    )
    assert ap._oa_get("authors", {"search": "x"}) == {}


def test_author_profile_oa_get_returns_data_on_success(monkeypatch):
    from paperscope.analysis import author_profile as ap

    monkeypatch.setattr(
        ap.openalex_client, "get",
        lambda *a, **k: oac.OpenAlexResult(ok=True, data={"results": [1]}),
    )
    assert ap._oa_get("authors") == {"results": [1]}


# ---------------------------------------------------------------------------
# related_radar: outage vs empty
# ---------------------------------------------------------------------------


def test_related_radar_surfaces_network_error(monkeypatch):
    from paperscope.analysis import related_radar as rr

    monkeypatch.setattr(
        rr, "_search_openalex",
        lambda *a, **k: oac.OpenAlexResult(ok=False, network_error=True, error="boom"),
    )
    result = rr.related_radar(r"\title{Perovskite Solar Cell Degradation Biomarkers}", n_results=10)
    assert result["network_error"] is True
    assert result["error"] == "OpenAlex unreachable"


def test_related_radar_genuine_empty_is_not_network_error(monkeypatch):
    from paperscope.analysis import related_radar as rr

    monkeypatch.setattr(
        rr, "_search_openalex",
        lambda *a, **k: oac.OpenAlexResult(ok=True, data={"results": []}),
    )
    result = rr.related_radar(r"\title{Perovskite Solar Cell Degradation Biomarkers}", n_results=10)
    assert result["network_error"] is False
    assert result["error"] == "No results from OpenAlex"


# ---------------------------------------------------------------------------
# journal_targeting: outage vs empty
# ---------------------------------------------------------------------------


def _paper_tex() -> str:
    return (
        "\\begin{document}\n\n"
        "This paragraph has clearly more than ten words in it so that the "
        "journal fit routine finds a section to embed.\n\n"
        "\\end{document}"
    )


def test_journal_fit_surfaces_network_error(monkeypatch):
    from paperscope.analysis import journal_targeting as jt

    def fake_embed(texts, model=None, show_progress=False):
        return np.ones((len(texts), 4)), "tfidf-fake"

    monkeypatch.setattr(jt, "embed_texts", fake_embed)
    monkeypatch.setattr(
        jt.openalex_client, "get",
        lambda *a, **k: oac.OpenAlexResult(ok=False, network_error=True, error="boom"),
    )
    # "S..." skips journal-name resolution; the works fetch then hits the outage.
    result = jt.journal_fit(_paper_tex(), ["S12345"], n_per_journal=10)
    assert result["network_error"] is True
    entry = result["rankings"][0]
    assert entry["network_error"] is True
    assert entry["error"] == "OpenAlex unreachable"


def test_fetch_journal_abstracts_flags_outage(monkeypatch):
    from paperscope.analysis import journal_targeting as jt

    monkeypatch.setattr(
        jt.openalex_client, "get",
        lambda *a, **k: oac.OpenAlexResult(ok=False, network_error=True, error="boom"),
    )
    abstracts, net_err = jt._fetch_journal_abstracts("S12345", n=10)
    assert abstracts == []
    assert net_err is True


def test_fetch_journal_abstracts_genuine_empty(monkeypatch):
    from paperscope.analysis import journal_targeting as jt

    monkeypatch.setattr(
        jt.openalex_client, "get",
        lambda *a, **k: oac.OpenAlexResult(ok=True, data={"results": []}),
    )
    abstracts, net_err = jt._fetch_journal_abstracts("S12345", n=10)
    assert abstracts == []
    assert net_err is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
