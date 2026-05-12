"""
Synthesizer Agent — DSKG traversal-based literature review generation (§4.5.3).
Generates structured review with inline confidence annotations.
"""

from __future__ import annotations

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.knowledge_graph.dskg import DSKG
from src.knowledge_graph.graph_algorithms import detect_communities, get_cluster_summary
from src.models.data_models import (
    GeneratedReview,
    PipelineState,
    ReviewSection,
)
from src.stores.conflict_registry import ConflictRegistry
from src.stores.gap_store import GapStore
from src.utils.prompts import SYNTHESIS_PROMPT


class SynthesizerAgent(BaseAgent):
    agent_name = "Synthesizer"
    agent_role = "Structured literature review synthesis via DSKG traversal"
    agent_goal = "Generate a comprehensive, confidence-annotated literature review"

    def __init__(
        self,
        settings=None,
        dskg: DSKG = None,
        conflict_registry: ConflictRegistry = None,
        gap_store: GapStore = None,
    ):
        super().__init__(settings)
        self.dskg = dskg
        self.conflict_registry = conflict_registry
        self.gap_store = gap_store

    def run(self, state: PipelineState) -> PipelineState:
        """Generate the structured literature review."""
        self.log_action("Starting literature review synthesis")

        review = GeneratedReview(
            topic=state.topic,
            total_papers_analyzed=len(state.papers),
            total_claims_extracted=len(state.claims),
            total_contradictions_found=len(state.conflicts),
            total_gaps_identified=len(state.gaps),
        )

        # Step 1: Determine thematic sections
        themes = self._determine_themes(state)
        self.log_action("Themes identified", f"{len(themes)} sections")

        # Step 2: Generate each thematic section
        for theme_name, theme_claims in themes.items():
            try:
                section = self._generate_section(theme_name, theme_claims, state)
                if section:
                    review.sections.append(section)
            except Exception as e:
                logger.warning(f"Section generation failed for '{theme_name}': {e}")

        # Step 3: Generate the Contested Claims section
        review.contested_claims_section = self._generate_contested_claims(state)

        # Step 4: Generate Research Gaps section
        review.research_gaps_section = self._generate_gaps_section(state)

        # Step 5: Generate Temporal Narrative section
        review.temporal_narrative = self._generate_temporal_section(state)

        # Step 6: Generate Conclusion
        review.conclusion = self._generate_conclusion(state, review)

        state.review = review
        state.status = "synthesizing"

        self.log_action("Synthesis complete", f"{len(review.sections)} sections generated")
        return state

    def _determine_themes(self, state: PipelineState) -> dict[str, list]:
        """Determine thematic sections from subtopics or DSKG communities."""
        themes: dict[str, list] = {}

        # Use subtopics as primary themes
        if state.subtopics:
            for subtopic in state.subtopics:
                subtopic_lower = subtopic.lower()
                matching_claims = []
                for claim_id, graded_claim in state.claims.items():
                    # Simple text matching for theme assignment
                    claim_text = graded_claim.claim.claim_text.lower()
                    entities = [e.lower() for e in graded_claim.claim.subject_entities]
                    if (
                        subtopic_lower in claim_text
                        or any(subtopic_lower in e for e in entities)
                        or any(e in subtopic_lower for e in entities if len(e) > 3)
                    ):
                        matching_claims.append((claim_id, graded_claim))

                if matching_claims:
                    themes[subtopic] = matching_claims

        # If few themes found, create a general theme with remaining claims
        assigned_claim_ids = {cid for claims in themes.values() for cid, _ in claims}
        unassigned = [
            (cid, c) for cid, c in state.claims.items()
            if cid not in assigned_claim_ids
        ]
        if unassigned:
            themes["General Findings"] = unassigned

        # Ensure at least one theme
        if not themes:
            themes[state.topic] = list(state.claims.items())

        return themes

    def _generate_section(
        self,
        theme: str,
        theme_claims: list,
        state: PipelineState,
    ) -> ReviewSection | None:
        """Generate a single thematic review section."""
        # Build claims with confidence for the prompt
        claims_str = "\n".join(
            f"- [{gc.confidence_category.value.upper()}] {gc.claim.claim_text[:200]} "
            f"(Paper: {gc.claim.source_paper_id})"
            for _, gc in theme_claims[:20]
        )

        # Find relevant contradictions
        theme_claim_ids = {cid for cid, _ in theme_claims}
        relevant_conflicts = [
            c for c in state.conflicts
            if c.pair.claim_a_id in theme_claim_ids or c.pair.claim_b_id in theme_claim_ids
        ]
        conflicts_str = "\n".join(
            f"- {c.pair.claim_a_text[:100]} VS {c.pair.claim_b_text[:100]} "
            f"(Type: {c.conflict_type.value})"
            for c in relevant_conflicts[:5]
        ) or "No contradictions in this theme."

        # Find relevant gaps
        theme_lower = theme.lower()
        relevant_gaps = [
            g for g in state.gaps
            if theme_lower in g.topic_cluster.lower() or theme_lower in g.gap_statement.lower()
        ]
        gaps_str = "\n".join(
            f"- {g.gap_statement[:200]} (confidence: {g.confidence:.2f})"
            for g in relevant_gaps[:3]
        ) or "No specific gaps identified for this theme."

        # Find relevant temporal narrative
        temporal_str = ""
        for thread in state.temporal_threads:
            if theme_lower in thread.topic.lower() or thread.topic.lower() in theme_lower:
                temporal_str = thread.narrative[:500] if thread.narrative else ""
                break

        prompt = SYNTHESIS_PROMPT.format(
            theme=theme,
            claims_with_confidence=claims_str,
            contradictions=conflicts_str,
            gaps=gaps_str,
            temporal_narrative=temporal_str or "No temporal analysis available.",
        )

        system = (
            "You are an expert academic writer synthesizing a comprehensive literature review. "
            "Write in a scholarly but accessible style. Integrate findings across papers "
            "rather than summarizing individual papers. Return valid JSON."
        )

        result = self.invoke_json(prompt, system)

        if not result:
            return None

        return ReviewSection(
            title=result.get("section_title", theme),
            content=result.get("content", ""),
            theme_cluster=theme,
            claim_ids=[cid for cid, _ in theme_claims],
        )

    def _generate_contested_claims(self, state: PipelineState) -> str:
        """Generate the 'Contested Claims and Methodological Disputes' section."""
        if self.conflict_registry:
            return self.conflict_registry.to_narrative()

        if not state.conflicts:
            return "No significant contradictions were identified in the reviewed literature."

        # Generate from state
        sections = []
        sections.append("## Contested Claims and Methodological Disputes\n")

        for conflict in state.conflicts:
            sections.append(f"**{conflict.conflict_id}** ({conflict.conflict_type.value})")
            sections.append(f"- Claim A: {conflict.pair.claim_a_text[:200]}")
            sections.append(f"- Claim B: {conflict.pair.claim_b_text[:200]}")
            sections.append(f"- Reconciliation: {conflict.reconciliation_statement[:300]}\n")

        return "\n".join(sections)

    def _generate_gaps_section(self, state: PipelineState) -> str:
        """Generate the Research Gaps section."""
        if self.gap_store:
            return self.gap_store.to_narrative()

        if not state.gaps:
            return "No significant research gaps were identified."

        sections = ["## Identified Research Gaps\n"]
        for gap in sorted(state.gaps, key=lambda g: g.confidence, reverse=True):
            sections.append(f"**{gap.gap_id}** ({gap.gap_type.value}, confidence: {gap.confidence:.2f})")
            sections.append(f"- {gap.gap_statement}")
            if gap.falsifiability:
                sections.append(f"- Falsifiability: {gap.falsifiability}")
            sections.append("")

        return "\n".join(sections)

    def _generate_temporal_section(self, state: PipelineState) -> str:
        """Generate the Temporal Evolution section."""
        if not state.temporal_threads:
            return ""

        sections = ["## Temporal Evolution of the Field\n"]
        for thread in state.temporal_threads:
            if thread.narrative:
                sections.append(f"### {thread.topic.title()}")
                sections.append(f"*Trajectory: {thread.trajectory_type}*\n")
                sections.append(thread.narrative)
                sections.append("")

        return "\n".join(sections)

    def _generate_conclusion(self, state: PipelineState, review: GeneratedReview) -> str:
        """Generate the conclusion section."""
        prompt = f"""Generate a concise conclusion for a literature review on "{state.topic}".

Key statistics:
- Papers analyzed: {review.total_papers_analyzed}
- Claims extracted: {review.total_claims_extracted}
- Contradictions found: {review.total_contradictions_found}
- Research gaps identified: {review.total_gaps_identified}

Section titles covered: {', '.join(s.title for s in review.sections)}

Write 2-3 paragraphs summarizing the state of the field, key findings, and directions for future work."""

        try:
            result = self.invoke(prompt)
            return result
        except Exception as e:
            logger.warning(f"Conclusion generation failed: {e}")
            return (
                f"This review analyzed {review.total_papers_analyzed} papers on {state.topic}, "
                f"extracting {review.total_claims_extracted} claims, identifying "
                f"{review.total_contradictions_found} contradictions and "
                f"{review.total_gaps_identified} research gaps."
            )
