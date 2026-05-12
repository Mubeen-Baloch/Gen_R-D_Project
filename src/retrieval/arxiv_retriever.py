"""
arXiv API client for preprint retrieval.
Uses the official arxiv Python package for search and metadata.
"""

from __future__ import annotations

from typing import Optional

import arxiv
from loguru import logger

from src.models.data_models import Paper


class ArxivRetriever:
    """Client for searching and retrieving papers from arXiv."""

    def __init__(self):
        self.client = arxiv.Client(
            page_size=20,
            delay_seconds=3.0,
            num_retries=3,
        )

    def search_papers(
        self,
        query: str,
        max_results: int = 20,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    ) -> list[Paper]:
        """
        Search arXiv for papers matching a query.
        
        Args:
            query: Search query (supports arXiv query syntax)
            max_results: Maximum number of results
            sort_by: Sort criterion (Relevance, LastUpdatedDate, SubmittedDate)
            
        Returns:
            List of Paper objects
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
        )

        papers = []
        try:
            for result in self.client.results(search):
                paper = self._parse_result(result)
                if paper:
                    papers.append(paper)
        except Exception as e:
            logger.error(f"arXiv search failed for '{query}': {e}")

        logger.info(f"Retrieved {len(papers)} papers from arXiv for: '{query[:50]}...'")
        return papers

    def get_paper_by_id(self, arxiv_id: str) -> Optional[Paper]:
        """Fetch a single paper by its arXiv ID (e.g., '2301.00001')."""
        search = arxiv.Search(id_list=[arxiv_id])
        try:
            results = list(self.client.results(search))
            if results:
                return self._parse_result(results[0])
        except Exception as e:
            logger.error(f"Failed to fetch arXiv paper {arxiv_id}: {e}")
        return None

    def download_pdf(self, arxiv_id: str, download_dir: str) -> Optional[str]:
        """Download a paper's PDF to the specified directory."""
        search = arxiv.Search(id_list=[arxiv_id])
        try:
            results = list(self.client.results(search))
            if results:
                path = results[0].download_pdf(dirpath=download_dir)
                logger.info(f"Downloaded PDF for {arxiv_id} to {path}")
                return str(path)
        except Exception as e:
            logger.error(f"Failed to download PDF for {arxiv_id}: {e}")
        return None

    def _parse_result(self, result: arxiv.Result) -> Optional[Paper]:
        """Convert an arxiv.Result to a Paper model."""
        try:
            # Extract year from published date
            year = result.published.year if result.published else 0

            # Extract arXiv ID from entry_id
            arxiv_id = result.entry_id.split("/")[-1] if result.entry_id else ""

            return Paper(
                paper_id=f"arxiv-{arxiv_id}",
                title=result.title or "",
                authors=[a.name for a in (result.authors or [])],
                year=year,
                venue="arXiv",
                doi=result.doi or "",
                arxiv_id=arxiv_id,
                abstract=result.summary or "",
                url=result.entry_id or "",
                pdf_path="",
                citation_count=0,
            )
        except Exception as e:
            logger.warning(f"Failed to parse arXiv result: {e}")
            return None
