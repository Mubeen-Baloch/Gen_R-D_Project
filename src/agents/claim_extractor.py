"""
Claim Extractor Agent — extracts atomic scientific claims from papers (§4.2.1, §5.2.1).
Processes each paper section through structured prompts to decompose into
method, result, and theoretical claims.
"""

from __future__ import annotations

from loguru import logger
from tqdm import tqdm

from src.agents.base_agent import BaseAgent
from src.models.data_models import (
    AtomicClaim,
    ClaimType,
    ConfidenceGradedClaim,
    PipelineState,
)
from src.utils.prompts import CLAIM_EXTRACTION_PROMPT


class ClaimExtractorAgent(BaseAgent):
    agent_name = "ClaimExtractor"
    agent_role = "Atomic scientific claim extraction from research papers"
    agent_goal = "Extract all atomic propositional claims from each paper"

    def run(self, state: PipelineState) -> PipelineState:
        """Extract claims from all processed papers in the corpus."""
        self.log_action("Starting claim extraction", f"{len(state.papers)} papers")

        processed_papers = [
            p for p in state.papers.values()
            if p.is_processed and p.sections
        ]

        if not processed_papers:
            # Fallback: extract from abstracts for papers without full processing
            processed_papers = [p for p in state.papers.values() if p.abstract]
            self.log_action("Using abstracts for claim extraction", f"{len(processed_papers)} papers")

        total_claims = 0

        for paper in tqdm(processed_papers, desc="Extracting claims"):
            try:
                claims = self._extract_from_paper(paper)
                for claim in claims:
                    graded_claim = ConfidenceGradedClaim(claim=claim)
                    state.claims[claim.claim_id] = graded_claim
                    total_claims += 1
            except Exception as e:
                logger.warning(f"Claim extraction failed for '{paper.title[:50]}': {e}")
                state.errors.append(f"Claim extraction failed for {paper.paper_id}: {e}")

        self.log_action("Claim extraction complete", f"{total_claims} claims from {len(processed_papers)} papers")
        return state

    def _extract_from_paper(self, paper) -> list[AtomicClaim]:
        """Extract atomic claims from a single paper."""
        all_claims = []

        # Prioritize key sections; fall back to abstract
        sections_to_process = []
        for section in paper.sections:
            if section.section_type in ("abstract", "introduction", "methods", "results", "discussion", "conclusion"):
                sections_to_process.append(section)

        if not sections_to_process and paper.abstract:
            from src.models.data_models import Section
            sections_to_process = [
                Section(title="Abstract", content=paper.abstract, section_type="abstract")
            ]

        for section in sections_to_process:
            if not section.content or len(section.content.strip()) < 50:
                continue

            # Truncate very long sections to stay within context limits
            content = section.content[:4000]

            prompt = CLAIM_EXTRACTION_PROMPT.format(
                paper_title=paper.title,
                paper_year=paper.year,
                section_type=section.section_type,
                section_content=content,
            )

            system = (
                "You are a precise scientific claim extractor. "
                "Extract ONLY claims that are explicitly stated or strongly implied in the text. "
                "Do not hallucinate or infer claims not supported by the text. "
                "Return valid JSON."
            )

            try:
                result = self.invoke_json(prompt, system)
                claims_data = result.get("claims", [])

                for cd in claims_data:
                    if not isinstance(cd, dict) or not cd.get("claim_text"):
                        continue

                    # Map claim type
                    claim_type_str = cd.get("claim_type", "theoretical").lower()
                    try:
                        claim_type = ClaimType(claim_type_str)
                    except ValueError:
                        claim_type = ClaimType.THEORETICAL

                    claim = AtomicClaim(
                        claim_text=cd["claim_text"],
                        claim_type=claim_type,
                        source_paper_id=paper.paper_id,
                        source_section=section.section_type,
                        confidence_indicators=cd.get("confidence_indicators", []),
                        subject_entities=cd.get("subject_entities", []),
                        condition_qualifiers=cd.get("condition_qualifiers", []),
                    )
                    all_claims.append(claim)

            except Exception as e:
                logger.debug(
                    f"Claim extraction failed for section '{section.section_type}' "
                    f"of '{paper.title[:40]}': {e}"
                )

        return all_claims
