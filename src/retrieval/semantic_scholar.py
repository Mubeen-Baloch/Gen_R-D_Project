"""
Semantic Scholar API client for academic paper retrieval.
Provides search, metadata fetching, citation graph, and reference extraction.
"""

from __future__ import annotations

import time
from typing import Optional

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.data_models import Paper


BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

PAPER_FIELDS = (
    "paperId,title,abstract,year,venue,authors,citationCount,"
    "externalIds,url,references,openAccessPdf"
)


class SemanticScholarClient:
    """Client for the Semantic Scholar Academic Graph API."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers["x-api-key"] = api_key
        self.session.headers["Accept"] = "application/json"
        self._last_request_time = 0
        self._min_interval = 1.0 if not api_key else 0.2  # Rate limiting

    def _throttle(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def search_papers(
        self,
        query: str,
        limit: int = 20,
        year_range: str = "",
        offset: int = 0,
    ) -> list[Paper]:
        """
        Search for papers matching a query.
        
        Args:
            query: Search query string
            limit: Max results (up to 100)
            year_range: e.g., "2020-2025"
            offset: Pagination offset
            
        Returns:
            List of Paper objects with metadata
        """
        self._throttle()
        params = {
            "query": query,
            "limit": min(limit, 100),
            "offset": offset,
            "fields": PAPER_FIELDS,
        }
        if year_range:
            params["year"] = year_range

        try:
            response = self.session.get(SEARCH_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []

        papers = []
        for item in data.get("data", []):
            paper = self._parse_paper(item)
            if paper:
                papers.append(paper)

        logger.info(f"Retrieved {len(papers)} papers for query: '{query[:50]}...'")
        return papers

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Fetch a single paper by its Semantic Scholar ID or DOI."""
        self._throttle()
        url = f"{BASE_URL}/paper/{paper_id}"
        try:
            response = self.session.get(
                url, params={"fields": PAPER_FIELDS}, timeout=30
            )
            response.raise_for_status()
            return self._parse_paper(response.json())
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch paper {paper_id}: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def get_references(self, paper_id: str, limit: int = 50) -> list[Paper]:
        """Get papers referenced by the given paper."""
        self._throttle()
        url = f"{BASE_URL}/paper/{paper_id}/references"
        try:
            response = self.session.get(
                url,
                params={"fields": PAPER_FIELDS, "limit": limit},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            papers = []
            for item in data.get("data", []):
                cited = item.get("citedPaper", {})
                if cited:
                    paper = self._parse_paper(cited)
                    if paper:
                        papers.append(paper)
            return papers
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch references for {paper_id}: {e}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def get_citations(self, paper_id: str, limit: int = 50) -> list[Paper]:
        """Get papers that cite the given paper."""
        self._throttle()
        url = f"{BASE_URL}/paper/{paper_id}/citations"
        try:
            response = self.session.get(
                url,
                params={"fields": PAPER_FIELDS, "limit": limit},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            papers = []
            for item in data.get("data", []):
                citing = item.get("citingPaper", {})
                if citing:
                    paper = self._parse_paper(citing)
                    if paper:
                        papers.append(paper)
            return papers
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch citations for {paper_id}: {e}")
            return []

    def _parse_paper(self, data: dict) -> Optional[Paper]:
        """Parse API response into a Paper model."""
        if not data or not data.get("title"):
            return None

        external_ids = data.get("externalIds", {}) or {}
        authors_list = data.get("authors", []) or []
        references = data.get("references", []) or []
        open_access = data.get("openAccessPdf", {}) or {}

        return Paper(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            authors=[a.get("name", "") for a in authors_list if a],
            year=data.get("year") or 0,
            venue=data.get("venue", "") or "",
            doi=external_ids.get("DOI", ""),
            arxiv_id=external_ids.get("ArXiv", ""),
            abstract=data.get("abstract", "") or "",
            url=data.get("url", "") or "",
            pdf_path="",
            citation_count=data.get("citationCount", 0) or 0,
            references=[r.get("paperId", "") for r in references if r and r.get("paperId")],
        )
