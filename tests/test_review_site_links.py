"""Link-integrity regression for the static review site (`ui.build`).

Builds a small site into tmp_path, parses every href in every emitted HTML
file, and asserts each internal target exists on disk. Guards the two v0
dead-link modes: decision-list pages linking `record/<pmid>.html` relative to
`screening/` (must be `../record/`), and the index linking
`screening/<decision>.html` for decision values whose list page is never
written (only include/exclude/maybe pages are emitted).

  python tests/test_review_site_links.py    # standalone, prints PASS/FAIL
  pytest tests/test_review_site_links.py    # under pytest
"""

from __future__ import annotations

import json
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from paperscope.systematic_review.ui import build_review_site  # noqa: E402


RECORDS = [
    {"pmid": "101", "title": "Included study", "abstract": "A", "journal": "J1", "year": "2020"},
    {"pmid": "102", "title": "Excluded study", "abstract": "B", "journal": "J2", "year": "2021"},
    {"pmid": "103", "title": "Maybe study", "abstract": "C", "journal": "J3", "year": "2022"},
    {"pmid": "104", "title": "Oddly screened study", "abstract": "D", "journal": "J4", "year": "2023"},
]
SCREENING = [
    {"pmid": "101", "decision": "include", "themes": ["dx"], "reason": ""},
    {"pmid": "102", "decision": "exclude", "themes": [], "reason": "review article"},
    {"pmid": "103", "decision": "maybe", "themes": [], "reason": "unclear"},
    # Non-standard decision value: counted on the index, but no list page is
    # ever written for it — the index must not link one.
    {"pmid": "104", "decision": "unsure", "themes": [], "reason": ""},
]


def _write_corpus(corpus: Path) -> None:
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in RECORDS) + "\n", encoding="utf-8"
    )
    (corpus / "screening.jsonl").write_text(
        "\n".join(json.dumps(d) for d in SCREENING) + "\n", encoding="utf-8"
    )


class _HrefCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.hrefs.append(v)


def _internal_hrefs(html_path: Path) -> list[str]:
    p = _HrefCollector()
    p.feed(html_path.read_text(encoding="utf-8"))
    return [h for h in p.hrefs if not h.startswith(("http://", "https://", "mailto:", "#"))]


def _dead_links(out: Path) -> tuple[int, list[str]]:
    """Return (n_links_checked, list of 'page -> href' dead links)."""
    dead: list[str] = []
    checked = 0
    for page in sorted(out.rglob("*.html")):
        for href in _internal_hrefs(page):
            checked += 1
            target = (page.parent / href.split("#", 1)[0]).resolve()
            if not target.exists():
                dead.append(f"{page.relative_to(out)} -> {href}")
    return checked, dead


def test_every_internal_link_resolves(tmp_path):
    corpus = tmp_path / "corpus"
    out = tmp_path / "site"
    _write_corpus(corpus)
    build_review_site(corpus, out, name="Link check")

    pages = sorted(out.rglob("*.html"))
    assert pages, "site emitted no HTML pages"
    checked, dead = _dead_links(out)
    assert checked > 0, "no internal links found — collector broken?"
    assert not dead, "dead internal links:\n" + "\n".join(dead)


def test_decision_lists_link_every_record_page(tmp_path):
    corpus = tmp_path / "corpus"
    out = tmp_path / "site"
    _write_corpus(corpus)
    build_review_site(corpus, out, name="Link check")

    # Every include/exclude/maybe record got a record page…
    for pmid in ("101", "102", "103"):
        assert (out / "record" / f"{pmid}.html").exists()
    # …and its decision-list row points at it (resolvable from screening/).
    for decision, pmid in (("include", "101"), ("exclude", "102"), ("maybe", "103")):
        page = out / "screening" / f"{decision}.html"
        hrefs = _internal_hrefs(page)
        record_hrefs = [h for h in hrefs if h.endswith(f"{pmid}.html")]
        assert record_hrefs, f"{page.name} has no link to record {pmid}"
        for h in record_hrefs:
            assert (page.parent / h).resolve().exists(), f"{page.name} -> {h} is dead"


def test_index_counts_unwritten_decisions_without_linking_them(tmp_path):
    corpus = tmp_path / "corpus"
    out = tmp_path / "site"
    _write_corpus(corpus)
    stats = build_review_site(corpus, out, name="Link check")

    assert stats["by_decision"].get("unsure") == 1  # still counted…
    index = out / "index.html"
    assert not (out / "screening" / "unsure.html").exists()
    hrefs = _internal_hrefs(index)
    assert not any("unsure" in h for h in hrefs), (
        "index links screening/unsure.html which is never written"
    )
    # The count is still visible on the index even though it isn't a link.
    assert "unsure" in index.read_text(encoding="utf-8")


if __name__ == "__main__":
    import tempfile

    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)


def test_hostile_pmid_cannot_escape_site_dir(tmp_path):
    """A pmid containing path separators must not write outside the site
    (record filenames are sanitized), and the page's links must still
    resolve to the sanitized filename."""
    corpus = tmp_path / "corpus-hostile"
    corpus.mkdir(parents=True)
    records = [{"pmid": "../evil", "title": "T", "abstract": "a",
                "journal": "J", "year": "2020"}]
    screening = [{"pmid": "../evil", "decision": "include", "themes": [],
                  "reason": ""}]
    (corpus / "records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    (corpus / "screening.jsonl").write_text(
        "\n".join(json.dumps(d) for d in screening) + "\n", encoding="utf-8")
    out = tmp_path / "site-hostile"
    build_review_site(corpus, out)
    # nothing escaped the record/ directory
    assert not (out / "evil.html").exists()
    assert not (tmp_path / "evil.html").exists()
    html_files = list(out.rglob("*.html"))
    assert all(out in p.parents or p.parent == out for p in html_files)
    # the record page landed inside record/ and every link resolves
    assert list((out / "record").glob("*.html")), "record page was not written"
    _, dead = _dead_links(out)
    assert not dead, dead
