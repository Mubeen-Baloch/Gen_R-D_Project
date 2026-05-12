"""
Gap Analyzer Agent — FRGO formalization (§4.4, §5.2.3).
Uses DSKG structural analysis + LLM to generate typed, evidence-grounded gap objects.
"""

from __future__ import annotations

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.knowledge_graph.dskg import DSKG
from src.knowledge_graph.graph_algorithms import (
    compute_centrality,
    detect_communities,
    find_structural_gaps,
    get_cluster_summary,
)
from src.models.data_models import GapClass, GapObject, GapType, PipelineState
from src.stores.gap_store import GapStore
from src.utils.prompts import GAP_ANALYSIS_PROMPT


class GapAnalyzerAgent(BaseAgent):
    agent_name = "GapAnalyzer"
    agent_role = "Formalized Research Gap identification and ontology construction"
    agent_goal = "Generate structured, verifiable gap objects with falsifiability statements"

    def __init__(self, settings=None, dskg: DSKG = None, gap_store: GapStore = None):
        super().__init__(settings)
        self.dskg = dskg
        self.gap_store = gap_store

    def run(self, state: PipelineState) -> PipelineState:
        """Analyze the DSKG for structural gaps and formalize them as FRGO objects."""
        self.log_action("Starting gap analysis")

        if not self.dskg or self.dskg.graph.number_of_nodes() < 5:
            self.log_action("Insufficient DSKG nodes for gap analysis")
            # Generate gaps from claims alone using LLM
            gaps = self._generate_gaps_from_claims(state)
            state.gaps = gaps
            if self.gap_store:
                self.gap_store.add_gaps_batch(gaps)
            return state

        # Step 1: Run community detection
        communities = detect_communities(self.dskg.graph)
        cluster_summaries = get_cluster_summary(self.dskg.graph, communities)

        # Step 2: Compute centrality
        centrality = compute_centrality(self.dskg.graph)

        # Step 3: Find structural gap indicators
        structural_gaps = find_structural_gaps(self.dskg.graph, communities, centrality)

        self.log_action("Structural analysis complete", f"{len(structural_gaps)} gap indicators")

        # Step 4: Formalize gaps using LLM
        formalized_gaps = []
        for gap_indicator in structural_gaps:
            try:
                gap_objects = self._formalize_gap(gap_indicator, cluster_summaries, state)
                formalized_gaps.extend(gap_objects)
            except Exception as e:
                logger.debug(f"Gap formalization failed: {e}")

        # Step 5: Also generate LLM-based gaps for comprehensive coverage
        llm_gaps = self._generate_gaps_from_claims(state)
        formalized_gaps.extend(llm_gaps)

        # Deduplicate by gap statement similarity
        seen_statements: set[str] = set()
        unique_gaps = []
        for gap in formalized_gaps:
            # Simple dedup by first 100 chars of statement
            key = gap.gap_statement[:100].lower()
            if key not in seen_statements:
                seen_statements.add(key)
                unique_gaps.append(gap)

        state.gaps = unique_gaps

        if self.gap_store:
            self.gap_store.add_gaps_batch(unique_gaps)

        self.log_action("Gap analysis complete", f"{len(unique_gaps)} formalized gaps")
        return state

    def _formalize_gap(
        self,
        gap_indicator: dict,
        cluster_summaries: dict,
        state: PipelineState,
    ) -> list[GapObject]:
        """Formalize a structural gap indicator into FRGO gap objects using LLM."""
        gap_type = gap_indicator.get("type", "")

        # Build structural evidence description
        evidence = gap_indicator.get("evidence", "")

        # Get relevant cluster info
        cluster_claims = ""
        cross_cluster = ""

        if "clusters" in gap_indicator:
            c1, c2 = gap_indicator["clusters"]
            for cid in (c1, c2):
                if cid in cluster_summaries:
                    summary = cluster_summaries[cid]
                    cluster_claims += f"Cluster {cid}: concepts={summary.get('concepts', [])}, "
                    cluster_claims += f"methods={summary.get('methods', [])}\n"
        elif "cluster" in gap_indicator:
            cid = gap_indicator["cluster"]
            if cid in cluster_summaries:
                summary = cluster_summaries[cid]
                cluster_claims = f"Concepts: {summary.get('concepts', [])}, Methods: {summary.get('methods', [])}"

        prompt = GAP_ANALYSIS_PROMPT.format(
            topic_cluster=gap_indicator.get("topics", gap_indicator.get("method_name", "General")),
            structural_evidence=evidence,
            cluster_claims=cluster_claims or "Not available",
            cross_cluster_info=cross_cluster or "Sparse connections detected",
        )

        system = (
            "You are a research gap identification expert. Generate precise, "
            "evidence-grounded, falsifiable research gap objects. Return valid JSON."
        )

        result = self.invoke_json(prompt, system)
        gaps_data = result.get("gaps", [])

        gaps = []
        for gd in gaps_data:
            if not isinstance(gd, dict) or not gd.get("gap_statement"):
                continue

            try:
                gap_type_enum = GapType(gd.get("gap_type", "unexplored_intersection"))
            except ValueError:
                gap_type_enum = GapType.UNEXPLORED_INTERSECTION

            try:
                gap_class_enum = GapClass(gd.get("gap_class", "methodological"))
            except ValueError:
                gap_class_enum = GapClass.METHODOLOGICAL

            gap = GapObject(
                gap_type=gap_type_enum,
                topic_cluster=str(gd.get("topic_cluster", "")),
                missing_intersection=gd.get("missing_intersection", []),
                gap_statement=gd["gap_statement"],
                evidence_papers=gd.get("evidence_papers", []),
                evidence_type=gd.get("evidence_type", "structural_absence"),
                implied_by=gd.get("implied_by", []),
                confidence=float(gd.get("confidence", 0.5)),
                temporal_position=gd.get("temporal_position", ""),
                gap_class=gap_class_enum,
                falsifiability=gd.get("falsifiability", ""),
            )
            gaps.append(gap)

        return gaps

    def _generate_gaps_from_claims(self, state: PipelineState) -> list[GapObject]:
        """Generate gaps directly from claim analysis when DSKG is insufficient."""
        # Summarize claims by type
        claim_summaries = []
        for claim in list(state.claims.values())[:30]:  # Limit for prompt length
            claim_summaries.append(
                f"- [{claim.claim.claim_type.value}] {claim.claim.claim_text[:150]} "
                f"(paper: {claim.claim.source_paper_id}, confidence: {claim.confidence_category.value})"
            )

        # Summarize contradictions
        contradiction_summaries = []
        for conflict in state.conflicts[:10]:
            contradiction_summaries.append(
                f"- {conflict.pair.claim_a_text[:100]} VS {conflict.pair.claim_b_text[:100]}"
            )

        prompt = GAP_ANALYSIS_PROMPT.format(
            topic_cluster=state.topic,
            structural_evidence=f"Based on {len(state.claims)} extracted claims from {len(state.papers)} papers",
            cluster_claims="\n".join(claim_summaries) if claim_summaries else "No claims available",
            cross_cluster_info=(
                "\n".join(contradiction_summaries) if contradiction_summaries
                else "No contradictions identified"
            ),
        )

        system = (
            "You are a research gap identification expert. Generate precise, "
            "evidence-grounded, falsifiable research gap objects. Return valid JSON."
        )

        try:
            result = self.invoke_json(prompt, system)
            gaps_data = result.get("gaps", [])
            gaps = []
            for gd in gaps_data:
                if not isinstance(gd, dict) or not gd.get("gap_statement"):
                    continue
                try:
                    gap_type_enum = GapType(gd.get("gap_type", "unexplored_intersection"))
                except ValueError:
                    gap_type_enum = GapType.UNEXPLORED_INTERSECTION
                try:
                    gap_class_enum = GapClass(gd.get("gap_class", "methodological"))
                except ValueError:
                    gap_class_enum = GapClass.METHODOLOGICAL

                gap = GapObject(
                    gap_type=gap_type_enum,
                    topic_cluster=str(gd.get("topic_cluster", state.topic)),
                    missing_intersection=gd.get("missing_intersection", []),
                    gap_statement=gd["gap_statement"],
                    evidence_papers=gd.get("evidence_papers", []),
                    evidence_type=gd.get("evidence_type", "implicit_boundary"),
                    implied_by=gd.get("implied_by", []),
                    confidence=float(gd.get("confidence", 0.5)),
                    temporal_position=gd.get("temporal_position", ""),
                    gap_class=gap_class_enum,
                    falsifiability=gd.get("falsifiability", ""),
                )
                gaps.append(gap)
            return gaps
        except Exception as e:
            logger.warning(f"LLM gap generation failed: {e}")
            return []
