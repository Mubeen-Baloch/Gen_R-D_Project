"""
Temporal Knowledge Evolution Tracker (TKET) — §4.6.
Models diachronic trajectories, consensus shifts, method emergence, and claim qualifications.
"""

from __future__ import annotations

from collections import defaultdict

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.data_models import (
    PipelineState,
    TemporalDataPoint,
    TemporalThread,
)
from src.utils.prompts import TEMPORAL_ANALYSIS_PROMPT


class TemporalTrackerAgent(BaseAgent):
    agent_name = "TemporalTracker"
    agent_role = "Temporal Knowledge Evolution Tracking"
    agent_goal = "Model the diachronic trajectory of research threads"

    def run(self, state: PipelineState) -> PipelineState:
        """Track temporal evolution of research threads across the corpus."""
        self.log_action("Starting temporal analysis")

        # Group claims by year and topic
        claims_by_year = self._organize_by_year(state)
        topic_threads = self._identify_threads(state)

        temporal_threads = []
        for thread_topic, thread_claims in topic_threads.items():
            try:
                thread = self._analyze_thread(thread_topic, thread_claims, claims_by_year, state)
                if thread:
                    temporal_threads.append(thread)
            except Exception as e:
                logger.debug(f"Temporal analysis failed for thread '{thread_topic}': {e}")

        state.temporal_threads = temporal_threads
        self.log_action("Temporal analysis complete", f"{len(temporal_threads)} threads tracked")
        return state

    def _organize_by_year(self, state: PipelineState) -> dict[int, list[str]]:
        """Organize claim IDs by publication year."""
        by_year: dict[int, list[str]] = defaultdict(list)

        for claim_id, graded_claim in state.claims.items():
            paper = state.papers.get(graded_claim.claim.source_paper_id)
            year = paper.year if paper else 0
            if year > 0:
                by_year[year].append(claim_id)

        return dict(by_year)

    def _identify_threads(self, state: PipelineState) -> dict[str, list]:
        """
        Identify research threads by clustering claims around common entities.
        Uses subject_entities from claim extraction.
        """
        entity_claims: dict[str, list] = defaultdict(list)

        for claim_id, graded_claim in state.claims.items():
            for entity in graded_claim.claim.subject_entities:
                entity_lower = entity.lower().strip()
                if len(entity_lower) > 2:  # Skip very short entities
                    entity_claims[entity_lower].append((claim_id, graded_claim))

        # Keep threads with at least 3 claims
        threads = {}
        for entity, claims in entity_claims.items():
            if len(claims) >= 3:
                threads[entity] = claims

        # Limit to top 10 threads by claim count
        sorted_threads = sorted(threads.items(), key=lambda x: len(x[1]), reverse=True)
        return dict(sorted_threads[:10])

    def _analyze_thread(
        self,
        thread_topic: str,
        thread_claims: list,
        claims_by_year: dict,
        state: PipelineState,
    ) -> TemporalThread | None:
        """Analyze a single research thread for temporal evolution."""

        # Build chronological claim list
        chronological = []
        method_timeline: dict[str, int] = {}

        for claim_id, graded_claim in thread_claims:
            paper = state.papers.get(graded_claim.claim.source_paper_id)
            year = paper.year if paper else 0
            chronological.append({
                "year": year,
                "claim_text": graded_claim.claim.claim_text[:200],
                "claim_type": graded_claim.claim.claim_type.value,
                "confidence": graded_claim.confidence_category.value,
                "paper_id": graded_claim.claim.source_paper_id,
            })

            # Track method emergence
            if graded_claim.claim.claim_type.value == "method":
                for entity in graded_claim.claim.subject_entities:
                    if entity.lower() not in method_timeline and year > 0:
                        method_timeline[entity.lower()] = year

        # Sort chronologically
        chronological.sort(key=lambda x: x["year"])

        if not chronological:
            return None

        # Build temporal data points
        data_points = []
        years_seen = sorted(set(c["year"] for c in chronological if c["year"] > 0))

        for year in years_seen:
            year_claims = [c for c in chronological if c["year"] == year]
            data_points.append(TemporalDataPoint(
                year=year,
                claim_ids=[c.get("paper_id", "") for c in year_claims],
                paper_ids=list(set(c.get("paper_id", "") for c in year_claims)),
                dominant_claim=year_claims[0]["claim_text"] if year_claims else "",
                consensus_strength=len(year_claims) / max(1, len(chronological)),
            ))

        # Use LLM to generate the diachronic narrative
        claims_str = "\n".join(
            f"[{c['year']}] ({c['claim_type']}, {c['confidence']}): {c['claim_text']}"
            for c in chronological[:20]
        )

        method_str = "\n".join(
            f"- {method}: first appeared {year}"
            for method, year in sorted(method_timeline.items(), key=lambda x: x[1])
        )

        prompt = TEMPORAL_ANALYSIS_PROMPT.format(
            thread_topic=thread_topic,
            chronological_claims=claims_str or "No chronological data available",
            method_timeline=method_str or "No method timeline available",
        )

        system = (
            "You are a historian of science expert at tracking how research "
            "consensus evolves over time. Generate insightful temporal narratives. "
            "Return valid JSON."
        )

        try:
            result = self.invoke_json(prompt, system)

            thread = TemporalThread(
                topic=thread_topic,
                trajectory_type=result.get("trajectory_type", "stable"),
                data_points=data_points,
                method_emergence=method_timeline,
                claim_qualifications=result.get("claim_qualifications", []),
                narrative=result.get("narrative", ""),
            )
            return thread
        except Exception as e:
            logger.debug(f"LLM temporal analysis failed: {e}")
            # Return thread with data but no narrative
            return TemporalThread(
                topic=thread_topic,
                trajectory_type="unknown",
                data_points=data_points,
                method_emergence=method_timeline,
            )
