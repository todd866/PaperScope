"""Tests for magic-byte verification in the harvest OA downloader.

ARCHITECTURE.md promises magic-byte verification on every download. The harvest
downloader previously trusted content-type / .pdf-suffix and streamed straight
to disk, so a Cloudflare/publisher block page served from a `.pdf` URL landed as
`<title>.pdf` and poisoned the corpus. These tests pin the byte-sniff.

Network is mocked, so they run anywhere.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from paperscope.harvest.download.open_access import (
    OpenAccessDownloader,
    download_papers,
)
from paperscope.harvest.sources.base import Paper


_VALID_PDF = b"%PDF-1.4\n%minimal\n1 0 obj <</Type/Catalog>> endobj\n%%EOF\n"
_BLOCK_PAGE = b"<!doctype html><html><body>Just a moment... Cloudflare</body></html>"


def _paper(pdf_url: str = "https://publisher.example/article.pdf") -> Paper:
    return Paper(
        id="10.1/x",
        title="A Study Of Things",
        authors=["A. Author"],
        abstract="",
        pdf_url=pdf_url,
    )


def _response(*, status: int, body: bytes, content_type: str = "application/pdf"):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": content_type}
    resp.raise_for_status = MagicMock()
    resp.iter_content = lambda chunk_size=8192: [body]
    return resp


def test_html_block_page_from_pdf_url_is_not_written(tmp_path):
    """A .pdf URL that actually serves an HTML block page (pdf content-type)
    must be rejected by the magic-byte check and leave nothing on disk."""
    dl = OpenAccessDownloader()

    def fake_get(url, **kwargs):
        return _response(status=200, body=_BLOCK_PAGE)

    with patch.object(dl.session, "get", side_effect=fake_get):
        out = dl.download(_paper(), tmp_path)

    assert out is None
    assert list(tmp_path.iterdir()) == []  # no poisoned <title>.pdf


def test_real_pdf_is_written(tmp_path):
    dl = OpenAccessDownloader()

    def fake_get(url, **kwargs):
        return _response(status=200, body=_VALID_PDF)

    with patch.object(dl.session, "get", side_effect=fake_get):
        out = dl.download(_paper(), tmp_path)

    assert out is not None
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF-")


def test_download_papers_drops_block_pages(tmp_path):
    """Aggregate path: a block-page 'PDF' must not appear in the result map."""
    papers = [_paper("https://a.example/one.pdf"), _paper("https://b.example/two.pdf")]

    def fake_get(url, **kwargs):
        body = _VALID_PDF if "one" in url else _BLOCK_PAGE
        return _response(status=200, body=body)

    with patch.object(requests.Session, "get", side_effect=fake_get):
        got = download_papers(papers, tmp_path)

    # Both papers share a title -> same filename; only the valid one should land.
    assert len(got) == 1
    assert all(p.read_bytes().startswith(b"%PDF-") for p in got.values())
