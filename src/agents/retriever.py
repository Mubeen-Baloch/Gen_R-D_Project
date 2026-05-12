"""
Retriever Agent — orchestrates Semantic Scholar + arXiv to build paper corpus (§4.1).
"""

from __future__ import annotations

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.data_models import PipelineState, Paper
from src.retrieval.semantic_scholar import SemanticScholarClient
from src.retrieval.arxiv_retriever import ArxivRetriever
from src.retrieval.paper_downloader import PaperDownloader


class RetrieverAgent(BaseAgent):
    agent_name = "Retriever"
    agent_role = "Academic paper retrieval and corpus construction"
    agent_goal = "Build a comprehensive paper corpus covering all subtopics"

    def __init__(self, settings=None):
        super().__init__(settings)
        self.s2_client = SemanticScholarClient(api_key=self.settings.semantic_scholar_api_key)
        self.arxiv_client = ArxivRetriever()
        self.downloader = PaperDownloader(papers_dir=self.settings.papers_dir)

    def run(self, state: PipelineState) -> PipelineState:
        """Retrieve papers for all subtopics and build the corpus."""
        self.log_action("Starting paper retrieval", f"Topic: {state.refined_query}")

        total_target = self.settings.max_papers
        papers_per_subtopic = max(5, total_target // max(1, len(state.subtopics)))
        year_range = f"{self.settings.temporal_scope_start}-{self.settings.temporal_scope_end}"

        all_papers: dict[str, Paper] = {}
        seen_titles: set[str] = set()

        for subtopic in state.subtopics:
            self.log_action("Retrieving for subtopic", subtopic)

            # Search Semantic Scholar
            s2_papers = self._search_semantic_scholar(subtopic, papers_per_subtopic, year_range)
            for p in s2_papers:
                if p.title.lower() not in seen_titles and p.paper_id:
                    all_papers[p.paper_id] = p
                    seen_titles.add(p.title.lower())

            # Search arXiv for supplementary coverage
            arxiv_papers = self._search_arxiv(subtopic, max(3, papers_per_subtopic // 2))
            for p in arxiv_papers:
                if p.title.lower() not in seen_titles and p.paper_id:
                    all_papers[p.paper_id] = p
                    seen_titles.add(p.title.lower())

            if len(all_papers) >= total_target:
                break

        # Also search with the refined query for broad coverage
        if len(all_papers) < total_target:
            s2_extra = self._search_semantic_scholar(
                state.refined_query, total_target - len(all_papers), year_range
            )
            for p in s2_extra:
                if p.title.lower() not in seen_titles and p.paper_id:
                    all_papers[p.paper_id] = p
                    seen_titles.add(p.title.lower())

        state.papers = all_papers
        state.status = "retrieving"

        self.log_action("Retrieval complete", f"{len(all_papers)} unique papers found")

        # Download PDFs
        self.log_action("Downloading PDFs")
        papers_list = list(all_papers.values())
        download_results = self.downloader.download_batch(papers_list, max_downloads=total_target)

        downloaded_count = len(download_results)
        self.log_action("PDF download complete", f"{downloaded_count}/{len(papers_list)} PDFs obtained")

        # Update paper objects with PDF paths
        for paper_id, path in download_results.items():
            if paper_id in state.papers:
                state.papers[paper_id].pdf_path = path

        return state

    def _search_semantic_scholar(self, query: str, limit: int, year_range: str) -> list[Paper]:
        """Search Semantic Scholar with error handling."""
        try:
            return self.s2_client.search_papers(query, limit=limit, year_range=year_range)
        except Exception as e:
            logger.warning(f"Semantic Scholar search failed for '{query}': {e}")
            return []

    def _search_arxiv(self, query: str, limit: int) -> list[Paper]:
        """Search arXiv with error handling."""
        try:
            return self.arxiv_client.search_papers(query, max_results=limit)
        except Exception as e:
            logger.warning(f"arXiv search failed for '{query}': {e}")
            return []

    def targeted_retrieval(self, state: PipelineState, queries: list[str]) -> PipelineState:
        """
        Execute targeted retrieval for specific queries (used by autonomy loop
        when coverage is below threshold).
        """
        self.log_action("Targeted retrieval", f"{len(queries)} additional queries")

        seen_titles = {p.title.lower() for p in state.papers.values()}

        for query in queries:
            papers = self._search_semantic_scholar(query, 10, "")
            for p in papers:
                if p.title.lower() not in seen_titles and p.paper_id:
                    state.papers[p.paper_id] = p
                    seen_titles.add(p.title.lower())

        # Download new PDFs
        new_papers = [p for p in state.papers.values() if not p.pdf_path]
        if new_papers:
            self.downloader.download_batch(new_papers)

        return state
