"""
Resolution Agent — generates reconciliation statements for confirmed contradictions (§4.3.3).
"""

from __future__ import annotations

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.data_models import (
    ConflictObject,
    ConflictPair,
    ConflictSeverity,
    ConflictType,
    PipelineState,
)
from src.stores.conflict_registry import ConflictRegistry
from src.utils.prompts import RESOLUTION_PROMPT


class ResolutionAgent(BaseAgent):
    agent_name = "ResolutionAgent"
    agent_role = "Contradiction resolution and reconciliation statement generation"
    agent_goal = "Generate reasoned reconciliation statements for all confirmed contradictions"

    def __init__(self, settings=None, conflict_registry: ConflictRegistry = None):
        super().__init__(settings)
        self.conflict_registry = conflict_registry

    def run(self, state: PipelineState) -> PipelineState:
        """Process all confirmed contradiction pairs and generate reconciliation statements."""
        self.log_action("Starting conflict resolution")

        # Collect confirmation results from the Contradiction Detector
        confirmation_results = {}
        for msg in state.message_bus:
            if msg.sender == "ContradictionDetector" and "pair_id" in msg.payload:
                confirmation_results[msg.payload["pair_id"]] = msg.payload

        confirmed_pairs = [
            p for p in state.conflict_pairs
            if p.pair_id in confirmation_results
        ]

        self.log_action("Processing confirmations", f"{len(confirmed_pairs)} confirmed pairs")

        conflicts = []
        for pair in confirmed_pairs:
            try:
                conflict = self._resolve_conflict(pair, confirmation_results.get(pair.pair_id, {}), state)
                if conflict:
                    conflicts.append(conflict)
            except Exception as e:
                logger.warning(f"Resolution failed for pair {pair.pair_id}: {e}")

        state.conflicts = conflicts

        # Add to conflict registry
        if self.conflict_registry:
            self.conflict_registry.add_conflicts_batch(conflicts)

        self.log_action("Resolution complete", f"{len(conflicts)} conflicts resolved")
        return state

    def _resolve_conflict(
        self,
        pair: ConflictPair,
        confirmation: dict,
        state: PipelineState,
    ) -> ConflictObject | None:
        """Generate a full ConflictObject with reconciliation statement."""
        llm_result = confirmation.get("result", {})

        # Get paper context
        claim_a = state.claims.get(pair.claim_a_id)
        claim_b = state.claims.get(pair.claim_b_id)
        if not claim_a or not claim_b:
            return None

        paper_a = state.papers.get(claim_a.claim.source_paper_id)
        paper_b = state.papers.get(claim_b.claim.source_paper_id)

        # Parse the conflict type from Stage 2 result
        conflict_type_str = llm_result.get("conflict_type", "methodological")
        try:
            conflict_type = ConflictType(conflict_type_str)
        except ValueError:
            conflict_type = ConflictType.METHODOLOGICAL

        severity_str = llm_result.get("severity", "contradictory")
        try:
            severity = ConflictSeverity(severity_str)
        except ValueError:
            severity = ConflictSeverity.CONTRADICTORY

        # If we already have a reconciliation from Stage 2, use it
        existing_reconciliation = llm_result.get("reconciliation_statement", "")
        existing_explanation = llm_result.get("explanation", "")

        # Generate a more detailed reconciliation if needed
        if len(existing_reconciliation) < 100:
            prompt = RESOLUTION_PROMPT.format(
                paper_a_title=paper_a.title if paper_a else "Unknown",
                paper_a_year=paper_a.year if paper_a else "Unknown",
                claim_a_text=pair.claim_a_text,
                paper_b_title=paper_b.title if paper_b else "Unknown",
                paper_b_year=paper_b.year if paper_b else "Unknown",
                claim_b_text=pair.claim_b_text,
                conflict_type=conflict_type.value,
                explanation=existing_explanation or "No prior explanation.",
            )

            system = (
                "You are a senior researcher expert at resolving scientific disagreements. "
                "Generate nuanced, balanced reconciliation statements. Return valid JSON."
            )

            try:
                result = self.invoke_json(prompt, system)
                existing_reconciliation = result.get(
                    "reconciliation_statement", existing_reconciliation
                )
                if not existing_explanation:
                    existing_explanation = result.get("explanation", "")
            except Exception as e:
                logger.debug(f"Detailed resolution generation failed: {e}")

        return ConflictObject(
            pair=pair,
            severity=severity,
            conflict_type=conflict_type,
            explanation=existing_explanation,
            reconciliation_statement=existing_reconciliation,
            source_paper_a_id=claim_a.claim.source_paper_id,
            source_paper_b_id=claim_b.claim.source_paper_id,
        )
