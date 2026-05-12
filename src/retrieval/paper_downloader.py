"""
Paper downloader — fetches open-access PDFs via direct URLs and Unpaywall.
Manages local paper cache in data/papers/.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.data_models import Paper


UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
USER_EMAIL = "research-framework@example.com"  # Required by Unpaywall API


class PaperDownloader:
    """Downloads open-access PDFs and manages a local cache."""

    def __init__(self, papers_dir: str = "./data/papers"):
        self.papers_dir = Path(papers_dir)
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "UncertaintyAwareSynthesis/1.0 (research-framework@example.com)"
        )

    def download_paper(self, paper: Paper) -> Optional[str]:
        """
        Attempt to download the PDF for a paper.
        Tries: direct open-access URL → Unpaywall → arXiv.
        Returns the local file path if successful.
        """
        # Check cache first
        cached = self._get_cached_path(paper)
        if cached:
            logger.debug(f"Using cached PDF for '{paper.title[:50]}...'")
            return cached

        # Strategy 1: Direct open-access PDF URL (from Semantic Scholar)
        if paper.url and "pdf" in paper.url.lower():
            path = self._download_url(paper.url, paper)
            if path:
                return path

        # Strategy 2: Unpaywall (if DOI available)
        if paper.doi:
            path = self._try_unpaywall(paper)
            if path:
                return path

        # Strategy 3: arXiv (if arXiv ID available)
        if paper.arxiv_id:
            path = self._try_arxiv(paper)
            if path:
                return path

        logger.warning(f"Could not download PDF for '{paper.title[:60]}...'")
        return None

    def download_batch(self, papers: list[Paper], max_downloads: int = 50) -> dict[str, str]:
        """
        Download PDFs for a batch of papers.
        Returns mapping of paper_id -> local_path for successful downloads.
        """
        results = {}
        for i, paper in enumerate(papers[:max_downloads]):
            if i > 0 and i % 10 == 0:
                logger.info(f"Downloaded {i}/{min(len(papers), max_downloads)} papers")
            path = self.download_paper(paper)
            if path:
                results[paper.paper_id] = path
                paper.pdf_path = path
            time.sleep(0.5)  # Be polite to servers
        logger.info(f"Successfully downloaded {len(results)}/{min(len(papers), max_downloads)} PDFs")
        return results

    def _get_cached_path(self, paper: Paper) -> Optional[str]:
        """Check if PDF is already downloaded."""
        safe_name = self._safe_filename(paper)
        path = self.papers_dir / safe_name
        if path.exists() and path.stat().st_size > 1000:  # Non-trivial file
            return str(path)
        return None

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(max=5))
    def _download_url(self, url: str, paper: Paper) -> Optional[str]:
        """Download a PDF from a direct URL."""
        try:
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type and "octet-stream" not in content_type:
                return None

            path = self.papers_dir / self._safe_filename(paper)
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if path.stat().st_size > 1000:
                logger.debug(f"Downloaded: {paper.title[:50]}...")
                return str(path)
            else:
                path.unlink(missing_ok=True)
                return None
        except Exception as e:
            logger.debug(f"Direct download failed for {url}: {e}")
            return None

    def _try_unpaywall(self, paper: Paper) -> Optional[str]:
        """Try to find and download open-access PDF via Unpaywall."""
        try:
            url = f"{UNPAYWALL_BASE}/{paper.doi}?email={USER_EMAIL}"
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()

            # Try best open-access location
            best_oa = data.get("best_oa_location", {})
            if best_oa:
                pdf_url = best_oa.get("url_for_pdf") or best_oa.get("url")
                if pdf_url:
                    return self._download_url(pdf_url, paper)
        except Exception as e:
            logger.debug(f"Unpaywall lookup failed for DOI {paper.doi}: {e}")
        return None

    def _try_arxiv(self, paper: Paper) -> Optional[str]:
        """Download PDF from arXiv."""
        arxiv_id = paper.arxiv_id.replace("v", ".").split(".")[0] if "v" in paper.arxiv_id else paper.arxiv_id
        pdf_url = f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"
        return self._download_url(pdf_url, paper)

    def _safe_filename(self, paper: Paper) -> str:
        """Generate a safe filename for a paper PDF."""
        # Use paper_id as base, sanitize
        base = paper.paper_id or paper.arxiv_id or paper.doi or "unknown"
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)
        return f"{safe}.pdf"
