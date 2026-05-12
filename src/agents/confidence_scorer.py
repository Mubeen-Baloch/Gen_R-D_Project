"""
Confidence Scorer — implements CGCERS formula (§4.2.2).
Computes confidence scores based on consensus, recency, and contradiction.
"""

from __future__ import annotations

import math
from datetime import datetime

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.data_models import PipelineState, ConfidenceGradedClaim
from src.models.embeddings import EmbeddingManager
from src.stores.claim_store import ClaimStore


class ConfidenceScorerAgent(BaseAgent):
    agent_name = "ConfidenceScorer"
    agent_role = "Confidence-Graded Claim Extraction and Reliability Scoring"
    agent_goal = "Assign calibrated confidence scores to extracted claims"

    def __init__(self, settings=None, embedding_manager: EmbeddingManager = None, claim_store: ClaimStore = None):
        super().__init__(settings)
        self.embedding_manager = embedding_manager
        self.claim_store = claim_store

    def run(self, state: PipelineState) -> PipelineState:
        """Compute confidence scores for all claims in the pipeline state."""
        self.log_action("Starting confidence scoring", f"{len(state.claims)} claims")

        if not self.embedding_manager or not self.claim_store:
            logger.warning("EmbeddingManager or ClaimStore not initialized; skipping scoring")
            return state

        # Phase 1: Generate embeddings for all claims and add to store
        self.log_action("Generating embeddings")
        claims_list = list(state.claims.values())
        claim_texts = [c.claim.claim_text for c in claims_list]

        if not claim_texts:
            return state

        embeddings = self.embedding_manager.embed_texts(claim_texts)

        for claim, embedding in zip(claims_list, embeddings):
            claim.claim.embedding = embedding

        # Add all claims to store
        self.claim_store.add_claims_batch(claims_list, embeddings)

        # Phase 2: Compute confidence scores
        self.log_action("Computing confidence scores")
        current_year = datetime.now().year

        for claim in claims_list:
            self._score_claim(claim, state, current_year)

        # Summary stats
        categories = {}
        for c in claims_list:
            cat = c.confidence_category.value
            categories[cat] = categories.get(cat, 0) + 1

        self.log_action("Confidence scoring complete", f"Distribution: {categories}")
        return state

    def _score_claim(self, claim: ConfidenceGradedClaim, state: PipelineState, current_year: int):
        """Compute the CGCERS confidence score for a single claim."""
        if not claim.claim.embedding:
            return

        # ── Consensus Score ──────────────────────────────────────
        # Find semantically equivalent claims from other papers
        similar_claims = self.claim_store.find_similar_claims(
            query_embedding=claim.claim.embedding,
            n_results=20,
            threshold=self.settings.embedding_similarity_threshold,
            exclude_paper_id=claim.claim.source_paper_id,
        )

        supporting_papers = set()
        for sc in similar_claims:
            paper_id = sc.get("metadata", {}).get("source_paper_id", "")
            if paper_id and paper_id != claim.claim.source_paper_id:
                supporting_papers.add(paper_id)

        total_papers = len(state.papers)
        claim.consensus_score = len(supporting_papers) / max(1, total_papers - 1)
        claim.supporting_paper_ids = list(supporting_papers)

        # ── Recency Score ────────────────────────────────────────
        # Time-decay weighted score: sum of e^(-lambda * (current_year - paper_year))
        recency_scores = []
        source_year = state.papers.get(claim.claim.source_paper_id)
        if source_year:
            source_year_val = source_year.year
        else:
            source_year_val = current_year

        # Include source paper's recency
        decay = self.settings.recency_decay
        recency_scores.append(
            math.exp(-decay * (current_year - source_year_val))
        )

        # Include supporting papers' recency
        for paper_id in supporting_papers:
            paper = state.papers.get(paper_id)
            if paper and paper.year:
                recency_scores.append(
                    math.exp(-decay * (current_year - paper.year))
                )

        claim.recency_score = sum(recency_scores) / max(1, len(recency_scores))

        # ── Contradiction Score ──────────────────────────────────
        # This will be updated after contradiction detection runs
        # For now, set to 0
        claim.contradiction_score = 0.0

        # ── Compute overall confidence ──────────────────────────
        claim.compute_confidence(
            alpha=self.settings.alpha_consensus,
            beta=self.settings.beta_recency,
            gamma=self.settings.gamma_contradiction,
        )

    def update_contradiction_scores(self, state: PipelineState):
        """
        Called after contradiction detection to update the contradiction
        component of confidence scores.
        """
        self.log_action("Updating contradiction scores")

        # Count contradictions per claim
        contradiction_counts: dict[str, set[str]] = {}
        for conflict in state.conflicts:
            claim_a = conflict.pair.claim_a_id
            claim_b = conflict.pair.claim_b_id

            contradiction_counts.setdefault(claim_a, set()).add(
                conflict.source_paper_b_id
            )
            contradiction_counts.setdefault(claim_b, set()).add(
                conflict.source_paper_a_id
            )

        total_papers = len(state.papers)

        for claim_id, graded_claim in state.claims.items():
            contradicting = contradiction_counts.get(claim_id, set())
            graded_claim.contradiction_score = len(contradicting) / max(1, total_papers - 1)
            graded_claim.contradicting_paper_ids = list(contradicting)

            # Recompute overall confidence
            graded_claim.compute_confidence(
                alpha=self.settings.alpha_consensus,
                beta=self.settings.beta_recency,
                gamma=self.settings.gamma_contradiction,
            )
