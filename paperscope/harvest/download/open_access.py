"""Download PDFs from open access sources."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests

from ..sources.base import Paper


class OpenAccessDownloader:
    """Download PDFs from open access sources (arXiv, bioRxiv, etc.)."""

    RATE_LIMIT_DELAY = 1.0

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/pdf,*/*",
        })
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _sanitize_filename(self, title: str) -> str:
        clean = re.sub(r'[^\w\s-]', '', title)
        clean = re.sub(r'\s+', '_', clean)
        return clean[:80]

    def download(self, paper: Paper, output_dir: Path) -> Optional[Path]:
        if not paper.pdf_url:
            return None

        self._rate_limit()

        filename = f"{self._sanitize_filename(paper.title)}.pdf"
        output_path = output_dir / filename

        if output_path.exists():
            return output_path

        try:
            response = self.session.get(paper.pdf_url, timeout=60, stream=True)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and not paper.pdf_url.endswith(".pdf"):
                return None

            # Magic-byte verification (ARCHITECTURE.md promises this on every
            # download). A .pdf URL / pdf content-type is not enough: publishers
            # and Cloudflare routinely serve an HTML block page from a .pdf
            # endpoint. Sniff the leading bytes before writing so an HTML block
            # page never lands as <title>.pdf and poisons the corpus. Consistent
            # with the b"%PDF-" check in ingest/open_access.py and shadow_library.
            chunks = response.iter_content(chunk_size=8192)
            header = b""
            buffered: list[bytes] = []
            for chunk in chunks:
                if not chunk:
                    continue
                buffered.append(chunk)
                header += chunk
                if len(header) >= 5:
                    break
            if header[:5] != b"%PDF-":
                return None

            with open(output_path, "wb") as f:
                for chunk in buffered:
                    f.write(chunk)
                for chunk in chunks:
                    f.write(chunk)

            return output_path

        except requests.RequestException as e:
            print(f"    Failed to download {paper.title[:50]}...: {e}")
            return None


def download_papers(papers: List[Paper], output_dir: Path) -> Dict[str, Path]:
    """Download PDFs for all papers with open access URLs."""
    downloader = OpenAccessDownloader()
    downloaded = {}

    for paper in papers:
        if paper.pdf_url:
            path = downloader.download(paper, output_dir)
            if path:
                downloaded[paper.id] = path
                print(f"    Downloaded: {path.name}")

    return downloaded
