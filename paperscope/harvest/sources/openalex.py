"""OpenAlex API client for paper discovery.

OpenAlex is fully open, no auth required, generous rate limits (10 req/sec with polite pool).
https://docs.openalex.org/
"""

from datetime import datetime, timedelta
from typing import List, Optional
import requests

from .base import DiscoverySource, Paper
from ...net import openalex_client


class OpenAlexSource(DiscoverySource):
    """Discover papers via OpenAlex API."""

    BASE_URL = openalex_client.OPENALEX_BASE

    SKIP_SOURCES = {"zenodo", "ssrn", "osf preprints", "research square", "authorea", "preprints.org"}

    def __init__(self, email: Optional[str] = None):
        self.session = requests.Session()
        self.email = email or "paper-harvester@example.com"

    @property
    def name(self) -> str:
        return "openalex"

    def _search(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        per_page: int = 50,
        require_abstract: bool = True,
    ) -> List[dict]:
        params = {
            "search": query,
            "per_page": per_page,
            "sort": "publication_date:desc",
        }

        filters = []
        if from_date:
            filters.append(f"from_publication_date:{from_date.strftime('%Y-%m-%d')}")
        if require_abstract:
            filters.append("has_abstract:true")
        if filters:
            params["filter"] = ",".join(filters)

        resp = openalex_client.get(
            "works", params, email=self.email, session=self.session
        )
        if resp.network_error:
            print(f"  Warning: OpenAlex search failed for '{query}': {resp.error}")
        return resp.results

    def _search_by_author(self, author_name: str, from_date: Optional[datetime] = None) -> List[dict]:
        params = {
            "search": author_name,
            "per_page": 25,
            "sort": "publication_date:desc",
        }

        if from_date:
            params["filter"] = f"from_publication_date:{from_date.strftime('%Y-%m-%d')}"

        resp = openalex_client.get(
            "works", params, email=self.email, session=self.session
        )
        if resp.network_error:
            print(f"  Warning: OpenAlex author search failed for '{author_name}': {resp.error}")
        return resp.results

    def _is_quality_source(self, work: dict) -> bool:
        primary = work.get("primary_location") or {}
        source = primary.get("source") or {}
        source_name = (source.get("display_name") or "").lower()
        for skip in self.SKIP_SOURCES:
            if skip in source_name:
                return False
        doi = work.get("doi", "") or ""
        if "10.5281" in doi:
            return False
        return True

    def _keyword_in_text(self, keyword: str, title: str, abstract: str) -> bool:
        keyword_lower = keyword.lower()
        title_lower = title.lower()
        abstract_lower = abstract.lower()
        in_title = keyword_lower in title_lower
        count_in_abstract = abstract_lower.count(keyword_lower)
        in_abstract_early = keyword_lower in abstract_lower[:500]
        return in_title or count_in_abstract >= 2 or in_abstract_early

    def _parse_work(self, work: dict, matched_keyword: str = "", matched_author: str = "") -> Optional[Paper]:
        doi = work.get("doi", "").replace("https://doi.org/", "") if work.get("doi") else None
        openalex_id = work.get("id", "").split("/")[-1]
        paper_id = doi or f"openalex:{openalex_id}"

        title = work.get("title") or work.get("display_name") or "Untitled"

        authors = []
        for auth in work.get("authorships", []):
            name = auth.get("author", {}).get("display_name")
            if name:
                authors.append(name)

        abstract = openalex_client.reconstruct_abstract(work)

        pub_date = None
        pub_date_str = work.get("publication_date")
        if pub_date_str:
            try:
                pub_date = datetime.fromisoformat(pub_date_str)
            except ValueError:
                pass

        pdf_url = None
        oa = work.get("open_access", {})
        if oa.get("is_oa"):
            pdf_url = oa.get("oa_url")
        best_oa = work.get("best_oa_location", {})
        if best_oa and not pdf_url:
            pdf_url = best_oa.get("pdf_url") or best_oa.get("landing_page_url")

        return Paper(
            id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            published_date=pub_date,
            source=self.name,
            source_url=work.get("id", ""),
            pdf_url=pdf_url,
            matched_keywords=[matched_keyword] if matched_keyword else [],
            matched_authors=[matched_author] if matched_author else [],
        )

    def search(
        self,
        keywords: List[str],
        authors: List[str],
        max_results: int = 50,
        lookback_days: int = 7,
    ) -> List[Paper]:
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        seen_ids: set = set()
        papers: List[Paper] = []

        for keyword in keywords:
            if len(papers) >= max_results:
                break

            print(f"  Searching OpenAlex for: {keyword}")
            results = self._search(keyword, from_date=cutoff_date, per_page=30)

            for work in results:
                if not self._is_quality_source(work):
                    continue
                paper = self._parse_work(work, matched_keyword=keyword)
                if not paper or paper.id in seen_ids:
                    continue
                if not self._keyword_in_text(keyword, paper.title, paper.abstract):
                    continue

                seen_ids.add(paper.id)

                for auth in authors:
                    auth_lower = auth.lower()
                    for name in paper.authors:
                        if auth_lower in name.lower():
                            paper.matched_authors.append(auth)
                            break

                papers.append(paper)
                if len(papers) >= max_results:
                    break

        for author in authors:
            if len(papers) >= max_results:
                break

            print(f"  Searching OpenAlex for author: {author}")
            results = self._search_by_author(author, from_date=cutoff_date)

            for work in results:
                if not self._is_quality_source(work):
                    continue
                paper = self._parse_work(work, matched_author=author)
                if not paper or paper.id in seen_ids:
                    continue

                seen_ids.add(paper.id)
                papers.append(paper)
                if len(papers) >= max_results:
                    break

        return papers
