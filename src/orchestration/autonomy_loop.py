"""
Autonomy Loop — orchestrates the multi-agent framework (§4.7).
Manages state, invokes agents in sequence, and handles critic feedback iterations.
"""

from __future__ import annotations

import time

from loguru import logger

from src.config.settings import get_settings
from src.models.data_models import PipelineState
from src.knowledge_graph.dskg import DSKG
from src.models.embeddings import EmbeddingManager
from src.stores.claim_store import ClaimStore
from src.stores.conflict_registry import ConflictRegistry
from src.stores.gap_store import GapStore
from src.pipeline.pdf_processor import PDFProcessor

# Agents
from src.agents.goal_interpreter import GoalInterpreterAgent
from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent
from src.agents.claim_extractor import ClaimExtractorAgent
from src.agents.confidence_scorer import ConfidenceScorerAgent
from src.agents.contradiction_detector import ContradictionDetectorAgent
from src.agents.resolution_agent import ResolutionAgent
from src.agents.dskg_builder import DSKGBuilderAgent
from src.agents.gap_analyzer import GapAnalyzerAgent
from src.agents.temporal_tracker import TemporalTrackerAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.critic import CriticAgent


class AutonomyLoop:
    """
    Central controller for the multi-agent system.
    Executes the task DAG and handles iterative refinement based on Critic feedback.
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

        # Initialize shared components
        self.embedding_manager = EmbeddingManager(
            provider=self.settings.llm_provider,
            model_name=self.settings.embedding_model,
            api_key=self.settings.google_api_key,
        )
        self.pdf_processor = PDFProcessor(
            grobid_url=self.settings.grobid_server_url,
            use_grobid=self.settings.use_grobid,
        )

        # Initialize stores
        self.dskg = DSKG(persist_path=self.settings.dskg_path)
        self.claim_store = ClaimStore(persist_dir=self.settings.claim_store_path)
        self.conflict_registry = ConflictRegistry(persist_path=self.settings.conflict_registry_path)
        self.gap_store = GapStore(persist_path=self.settings.gap_store_path)

        # Initialize agents
        self.interpreter = GoalInterpreterAgent(self.settings)
        self.planner = PlannerAgent(self.settings)
        self.retriever = RetrieverAgent(self.settings)
        self.extractor = ClaimExtractorAgent(self.settings)
        self.scorer = ConfidenceScorerAgent(
            self.settings, self.embedding_manager, self.claim_store
        )
        self.detector = ContradictionDetectorAgent(self.settings, self.claim_store)
        self.resolver = ResolutionAgent(self.settings, self.conflict_registry)
        self.dskg_builder = DSKGBuilderAgent(self.settings, self.dskg)
        self.gap_analyzer = GapAnalyzerAgent(self.settings, self.dskg, self.gap_store)
        self.temporal_tracker = TemporalTrackerAgent(self.settings)
        self.synthesizer = SynthesizerAgent(
            self.settings, self.dskg, self.conflict_registry, self.gap_store
        )
        self.critic = CriticAgent(self.settings)

    def execute(self, topic: str) -> PipelineState:
        """Run the full autonomous pipeline for a given topic."""
        logger.info(f"Starting autonomy loop for topic: '{topic}'")
        start_time = time.time()

        # Initialize state
        state = PipelineState(topic=topic)

        # ── Phase 1: Planning ──
        state = self.interpreter.run(state)
        state = self.planner.run(state)

        # ── Autonomy Iteration Loop ──
        for iteration in range(1, self.settings.max_autonomy_iterations + 1):
            logger.info(f"=== Starting Iteration {iteration}/{self.settings.max_autonomy_iterations} ===")
            state.current_iteration = iteration

            state = self._execute_iteration(state)

            if state.status == "complete":
                logger.info(f"Pipeline completed successfully on iteration {iteration}.")
                break
            elif state.status == "needs_improvement" and iteration < self.settings.max_autonomy_iterations:
                logger.info("Review needs improvement. Refining...")
                state = self._refine_state(state)
            else:
                logger.warning("Max iterations reached or unexpected status.")
                break

        elapsed = time.time() - start_time
        logger.info(f"Autonomy loop finished in {elapsed:.2f} seconds.")
        return state

    def _execute_iteration(self, state: PipelineState) -> PipelineState:
        """Execute one full pass through the core pipeline."""
        
        # 1. Retrieval (if needed)
        if not state.papers:
            state = self.retriever.run(state)

        # 2. PDF Processing
        unprocessed = [p for p in state.papers.values() if not p.is_processed and p.pdf_path]
        if unprocessed:
            logger.info(f"Processing {len(unprocessed)} PDFs")
            self.pdf_processor.process_batch(unprocessed)
            state.status = "processing"

        # 3. Extraction & Scoring
        if not state.claims or state.current_iteration > 1:
            state = self.extractor.run(state)
            state = self.scorer.run(state)

        state.status = "analyzing"

        # 4. Conflict Analysis
        state = self.detector.run(state)
        state = self.resolver.run(state)
        
        # 5. Update confidence scores with contradiction data
        self.scorer.update_contradiction_scores(state)

        # 6. DSKG Construction
        state = self.dskg_builder.run(state)

        # 7. Advanced Analysis
        state = self.gap_analyzer.run(state)
        state = self.temporal_tracker.run(state)

        # 8. Synthesis & Evaluation
        state = self.synthesizer.run(state)
        state = self.critic.run(state)

        return state

    def _refine_state(self, state: PipelineState) -> PipelineState:
        """Apply targeted improvements based on Critic feedback."""
        last_eval = state.evaluations[-1] if state.evaluations else None
        if not last_eval:
            return state

        logger.info(f"Refining based on feedback: {last_eval.feedback[:100]}...")

        # If claim coverage is low, do targeted retrieval
        if last_eval.claim_coverage_score < 0.6:
            queries = []
            for directive in last_eval.improvement_directives:
                if "search" in directive.lower() or "retrieve" in directive.lower():
                    # Extract quoted query if present
                    import re
                    match = re.search(r"'(.*?)'|\"(.*?)\"", directive)
                    if match:
                        queries.append(match.group(1) or match.group(2))
            
            if not queries:
                # Fallback queries based on subtopics with fewest papers
                queries = state.subtopics[:2]
                
            state = self.retriever.targeted_retrieval(state, queries)

        # Clear existing review to force re-synthesis
        state.review = None
        
        return state
