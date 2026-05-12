"""
Contradiction Detector Agent — two-stage CDRA (§4.3, §5.2.2).
Stage 1: NLI model candidate filtering.
Stage 2: LLM confirmation + 4-type classification.
"""

from __future__ import annotations

from loguru import logger
from tqdm import tqdm

from src.agents.base_agent import BaseAgent
from src.models.data_models import (
    ConflictPair,
    ConflictSeverity,
    PipelineState,
)
from src.stores.claim_store import ClaimStore
from src.utils.prompts import CONTRADICTION_CONFIRMATION_PROMPT


class ContradictionDetectorAgent(BaseAgent):
    agent_name = "ContradictionDetector"
    agent_role = "Two-stage contradiction detection across scientific claims"
    agent_goal = "Identify conflicting claims across papers with high precision and recall"

    def __init__(self, settings=None, claim_store: ClaimStore = None):
        super().__init__(settings)
        self.claim_store = claim_store
        self._nli_model = None
        self._nli_tokenizer = None

    def _init_nli_model(self):
        """Initialize the pre-trained NLI model for Stage 1 filtering."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            model_name = "cross-encoder/nli-deberta-v3-base"
            self._nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._nli_model.eval()
            logger.info(f"NLI model loaded: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load NLI model: {e}. Using embedding similarity only.")

    def run(self, state: PipelineState) -> PipelineState:
        """Run two-stage contradiction detection on all claims."""
        self.log_action("Starting contradiction detection", f"{len(state.claims)} claims")

        if len(state.claims) < 2:
            self.log_action("Too few claims for contradiction detection")
            return state

        # Stage 1: Candidate pair identification
        candidate_pairs = self._stage1_candidate_identification(state)
        self.log_action("Stage 1 complete", f"{len(candidate_pairs)} candidate pairs")

        # Stage 2: LLM confirmation and classification
        confirmed_pairs = self._stage2_llm_confirmation(candidate_pairs, state)
        self.log_action("Stage 2 complete", f"{len(confirmed_pairs)} confirmed contradictions")

        state.conflict_pairs = candidate_pairs
        return state

    def _stage1_candidate_identification(self, state: PipelineState) -> list[ConflictPair]:
        """
        Stage 1: Identify candidate contradiction pairs using:
        1. Embedding similarity (high similarity = same topic)
        2. NLI contradiction scoring (if model available)
        """
        candidates = []
        claims_list = list(state.claims.values())
        seen_pairs: set[tuple[str, str]] = set()

        # Initialize NLI model if available
        if self._nli_model is None:
            self._init_nli_model()

        for claim in tqdm(claims_list, desc="Finding contradiction candidates"):
            if not claim.claim.embedding:
                continue

            # Find semantically similar claims from different papers
            similar = self.claim_store.find_contradiction_candidates(
                claim_embedding=claim.claim.embedding,
                claim_paper_id=claim.claim.source_paper_id,
                n_results=15,
                similarity_threshold=self.settings.embedding_similarity_threshold - 0.2,  # Lower threshold for candidates
            )

            for sim_claim in similar:
                pair_key = tuple(sorted([claim.claim.claim_id, sim_claim["claim_id"]]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Compute NLI contradiction score
                nli_score = 0.0
                if self._nli_model is not None:
                    nli_score = self._compute_nli_score(
                        claim.claim.claim_text,
                        sim_claim["claim_text"],
                    )
                else:
                    # Use similarity as proxy (high similarity + different papers = potential conflict)
                    nli_score = sim_claim["similarity"] * 0.5

                if nli_score >= self.settings.contradiction_score_threshold:
                    pair = ConflictPair(
                        claim_a_id=claim.claim.claim_id,
                        claim_b_id=sim_claim["claim_id"],
                        claim_a_text=claim.claim.claim_text,
                        claim_b_text=sim_claim["claim_text"],
                        embedding_similarity=sim_claim["similarity"],
                        nli_contradiction_score=nli_score,
                    )
                    candidates.append(pair)

        return candidates

    def _compute_nli_score(self, text_a: str, text_b: str) -> float:
        """
        Compute NLI contradiction probability for a claim pair.
        Uses cross-encoder NLI model (DeBERTa-v3).
        Returns probability that the pair is contradictory.
        """
        try:
            import torch

            inputs = self._nli_tokenizer(
                text_a[:512],
                text_b[:512],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            with torch.no_grad():
                outputs = self._nli_model(**inputs)
                logits = outputs.logits

            # NLI labels: 0=contradiction, 1=entailment, 2=neutral
            probs = torch.softmax(logits, dim=1)
            contradiction_prob = probs[0][0].item()

            return contradiction_prob
        except Exception as e:
            logger.debug(f"NLI scoring failed: {e}")
            return 0.0

    def _stage2_llm_confirmation(
        self, candidates: list[ConflictPair], state: PipelineState
    ) -> list[ConflictPair]:
        """
        Stage 2: LLM-based confirmation and classification.
        For each candidate pair, use the LLM to confirm and classify.
        """
        confirmed = []

        # Limit to top candidates to manage API costs
        sorted_candidates = sorted(
            candidates, key=lambda c: c.nli_contradiction_score, reverse=True
        )
        top_candidates = sorted_candidates[:50]  # Process top 50

        for pair in tqdm(top_candidates, desc="LLM contradiction confirmation"):
            # Get paper metadata for context
            claim_a = state.claims.get(pair.claim_a_id)
            claim_b = state.claims.get(pair.claim_b_id)

            if not claim_a or not claim_b:
                continue

            paper_a = state.papers.get(claim_a.claim.source_paper_id)
            paper_b = state.papers.get(claim_b.claim.source_paper_id)

            prompt = CONTRADICTION_CONFIRMATION_PROMPT.format(
                paper_a_title=paper_a.title if paper_a else "Unknown",
                paper_a_year=paper_a.year if paper_a else "Unknown",
                claim_a_text=pair.claim_a_text,
                paper_b_title=paper_b.title if paper_b else "Unknown",
                paper_b_year=paper_b.year if paper_b else "Unknown",
                claim_b_text=pair.claim_b_text,
            )

            system = (
                "You are an expert scientific reviewer analyzing potential contradictions. "
                "Be precise in your classification. Only classify as contradictory if the "
                "claims genuinely conflict, not merely if they discuss different aspects. "
                "Return valid JSON."
            )

            try:
                result = self.invoke_json(prompt, system)

                severity_str = result.get("severity", "non_contradictory")
                try:
                    severity = ConflictSeverity(severity_str)
                except ValueError:
                    severity = ConflictSeverity.NON_CONTRADICTORY

                if severity != ConflictSeverity.NON_CONTRADICTORY:
                    # Store the LLM classification result in the pair metadata
                    pair.nli_contradiction_score = max(
                        pair.nli_contradiction_score, 0.7
                    )
                    confirmed.append(pair)

                    # Store the full result for the Resolution Agent
                    from src.models.data_models import AgentMessage
                    state.message_bus.append(
                        AgentMessage(
                            sender=self.agent_name,
                            payload={
                                "pair_id": pair.pair_id,
                                "result": result,
                                "claim_a_id": pair.claim_a_id,
                                "claim_b_id": pair.claim_b_id,
                            },
                        )
                    )
            except Exception as e:
                logger.debug(f"LLM confirmation failed for pair {pair.pair_id}: {e}")

        return confirmed
