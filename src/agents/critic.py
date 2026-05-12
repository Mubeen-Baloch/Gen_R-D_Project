"""
Critic Agent — evaluates generated reviews and provides feedback (§4.7.3).
Returns evaluation metrics and improvement directives.
"""

from __future__ import annotations

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.data_models import CriticEvaluation, PipelineState
from src.utils.prompts import CRITIC_PROMPT


class CriticAgent(BaseAgent):
    agent_name = "Critic"
    agent_role = "Quality assurance and evaluation of generated literature reviews"
    agent_goal = "Evaluate reviews against 5 quality dimensions and provide improvement directives"

    def run(self, state: PipelineState) -> PipelineState:
        """Evaluate the synthesized literature review."""
        self.log_action("Starting evaluation of generated review")

        if not state.review:
            self.log_action("No review found to evaluate")
            return state

        review_text = ""
        for section in state.review.sections:
            review_text += f"## {section.title}\n{section.content}\n\n"

        review_text += state.review.contested_claims_section + "\n\n"
        review_text += state.review.research_gaps_section + "\n\n"
        review_text += state.review.temporal_narrative + "\n\n"
        review_text += f"## Conclusion\n{state.review.conclusion}"

        # Truncate if too long for prompt window
        if len(review_text) > 30000:
            review_text = review_text[:30000] + "\n...[truncated]..."

        prompt = CRITIC_PROMPT.format(
            topic=state.topic,
            num_papers=state.review.total_papers_analyzed,
            num_claims=state.review.total_claims_extracted,
            num_contradictions=state.review.total_contradictions_found,
            num_gaps=state.review.total_gaps_identified,
            review_text=review_text,
        )

        system = (
            "You are a rigorous, highly critical academic peer reviewer. "
            "Evaluate the literature review objectively against the criteria. "
            "Do not inflate scores. Identify specific, actionable flaws. "
            "Return valid JSON."
        )

        try:
            result = self.invoke_json(prompt, system)

            evaluation = CriticEvaluation(
                iteration=state.current_iteration,
                claim_coverage_score=float(result.get("claim_coverage_score", 0.5)),
                contradiction_completeness=float(result.get("contradiction_completeness", 0.5)),
                gap_formalization_quality=float(result.get("gap_formalization_quality", 0.5)),
                temporal_coherence_score=float(result.get("temporal_coherence_score", 0.5)),
                synthesis_coherence_score=float(result.get("synthesis_coherence_score", 0.5)),
                feedback=result.get("feedback", "No feedback provided."),
                improvement_directives=result.get("improvement_directives", []),
            )

            # Compute weighted overall score
            evaluation.compute_overall()

            state.evaluations.append(evaluation)

            self.log_action(
                "Evaluation complete",
                f"Overall score: {evaluation.overall_quality:.2f}"
            )

            # Check if it meets quality threshold
            if evaluation.overall_quality >= self.settings.quality_threshold:
                self.log_action("Review accepted", "Meets quality threshold")
                state.status = "complete"
            else:
                self.log_action(
                    "Review rejected",
                    f"Needs improvement. Feedback: {evaluation.feedback[:100]}..."
                )
                state.status = "needs_improvement"

        except Exception as e:
            logger.warning(f"Critic evaluation failed: {e}")
            state.status = "complete"  # Fail open if evaluation breaks

        return state
