"""
DSKG Builder Agent — assembles the Knowledge Graph from extracted claims and papers.
"""

from __future__ import annotations

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.knowledge_graph.dskg import DSKG
from src.models.data_models import PipelineState


class DSKGBuilderAgent(BaseAgent):
    agent_name = "DSKGBuilder"
    agent_role = "Knowledge Graph Construction"
    agent_goal = "Assemble papers, claims, and relations into the DSKG"

    def __init__(self, settings=None, dskg: DSKG = None):
        super().__init__(settings)
        self.dskg = dskg

    def run(self, state: PipelineState) -> PipelineState:
        """Construct the DSKG from the current state."""
        self.log_action("Starting DSKG construction")

        if not self.dskg:
            logger.warning("DSKG instance not provided; skipping construction")
            return state

        # Clear existing graph if rebuilding
        self.dskg.clear()

        # Add all papers
        for paper_id, paper in state.papers.items():
            self.dskg.add_paper_node(paper)

        # Add all claims
        for claim_id, graded_claim in state.claims.items():
            self.dskg.add_claim_node(graded_claim)

            # Add concepts/methods from subject_entities
            for entity in graded_claim.claim.subject_entities:
                if len(entity) > 2:
                    if graded_claim.claim.claim_type.value == "method":
                        self.dskg.add_method_node(entity, year=state.papers.get(graded_claim.claim.source_paper_id, paper).year)
                    else:
                        self.dskg.add_concept_node(entity)

        # Add support edges (consensus)
        support_edges_added = 0
        for claim_id, graded_claim in state.claims.items():
            for supporting_paper_id in graded_claim.supporting_paper_ids:
                # Find the specific claim in that paper
                # For simplicity, we just link to any claim from that paper that is similar
                # The ConfidenceScorer could be enhanced to store the exact supporting claim ID
                for other_id, other_claim in state.claims.items():
                    if other_id != claim_id and other_claim.claim.source_paper_id == supporting_paper_id:
                        # Assuming they are the similar ones
                        self.dskg.add_supports_edge(other_id, claim_id)
                        support_edges_added += 1
                        break

        # Add contradiction edges
        for conflict in state.conflicts:
            self.dskg.add_contradicts_edge(conflict.pair.claim_a_id, conflict.pair.claim_b_id)

        self.log_action(
            "DSKG construction complete",
            f"{self.dskg.graph.number_of_nodes()} nodes, {self.dskg.graph.number_of_edges()} edges"
        )

        # Save to disk
        self.dskg.save()

        return state
