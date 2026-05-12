"""
Gap Store — structured storage for FRGO gap objects (§4.4).
Persists to JSON and provides ranking/filtering/validation methods.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from src.models.data_models import GapObject, GapType, GapClass


class GapStore:
    """
    Storage for Formalized Research Gap Objects.
    Supports ranking, filtering, and empirical validation tracking.
    """

    def __init__(self, persist_path: str = "./data/gap_store.json"):
        self.persist_path = persist_path
        self.gaps: list[GapObject] = []
        self._load()

    def add_gap(self, gap: GapObject):
        """Add a gap object to the store."""
        existing_ids = {g.gap_id for g in self.gaps}
        if gap.gap_id not in existing_ids:
            self.gaps.append(gap)
            self._save()

    def add_gaps_batch(self, gaps: list[GapObject]):
        """Add multiple gaps at once."""
        existing_ids = {g.gap_id for g in self.gaps}
        added = 0
        for gap in gaps:
            if gap.gap_id not in existing_ids:
                self.gaps.append(gap)
                existing_ids.add(gap.gap_id)
                added += 1
        if added > 0:
            self._save()
            logger.info(f"Added {added} gaps to store (total: {len(self.gaps)})")

    def get_by_type(self, gap_type: GapType) -> list[GapObject]:
        """Filter gaps by type."""
        return [g for g in self.gaps if g.gap_type == gap_type]

    def get_by_class(self, gap_class: GapClass) -> list[GapObject]:
        """Filter gaps by class."""
        return [g for g in self.gaps if g.gap_class == gap_class]

    def get_top_k(self, k: int = 10) -> list[GapObject]:
        """Get top-K gaps ranked by confidence score."""
        sorted_gaps = sorted(self.gaps, key=lambda g: g.confidence, reverse=True)
        return sorted_gaps[:k]

    def get_by_cluster(self, cluster_name: str) -> list[GapObject]:
        """Get gaps belonging to a specific topic cluster."""
        return [g for g in self.gaps if g.topic_cluster.lower() == cluster_name.lower()]

    def get_stats(self) -> dict:
        """Get summary statistics."""
        type_counts = {}
        class_counts = {}
        for g in self.gaps:
            type_counts[g.gap_type.value] = type_counts.get(g.gap_type.value, 0) + 1
            class_counts[g.gap_class.value] = class_counts.get(g.gap_class.value, 0) + 1

        avg_confidence = (
            sum(g.confidence for g in self.gaps) / len(self.gaps)
            if self.gaps
            else 0.0
        )

        return {
            "total": len(self.gaps),
            "by_type": type_counts,
            "by_class": class_counts,
            "avg_confidence": avg_confidence,
        }

    def to_narrative(self) -> str:
        """Generate a narrative summary of research gaps for the review."""
        if not self.gaps:
            return "No significant research gaps were identified."

        sections = []
        sorted_gaps = sorted(self.gaps, key=lambda g: g.confidence, reverse=True)

        for gap_type in GapType:
            typed_gaps = [g for g in sorted_gaps if g.gap_type == gap_type]
            if not typed_gaps:
                continue

            type_label = gap_type.value.replace("_", " ").title()
            section = f"### {type_label} ({len(typed_gaps)} identified)\n\n"

            for gap in typed_gaps:
                section += f"**{gap.gap_id}** (confidence: {gap.confidence:.2f})\n"
                section += f"- *Statement*: {gap.gap_statement}\n"
                if gap.missing_intersection:
                    section += f"- *Missing intersection*: {', '.join(gap.missing_intersection)}\n"
                section += f"- *Evidence*: {'; '.join(gap.implied_by[:3])}\n"
                section += f"- *Falsifiability*: {gap.falsifiability}\n\n"

            sections.append(section)

        return "\n".join(sections)

    def count(self) -> int:
        return len(self.gaps)

    def clear(self):
        self.gaps = []
        self._save()

    def _save(self):
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        data = [g.model_dump(mode="json") for g in self.gaps]
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        path = Path(self.persist_path)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.gaps = [GapObject.model_validate(d) for d in data]
                logger.info(f"Loaded {len(self.gaps)} gaps from {self.persist_path}")
            except Exception as e:
                logger.warning(f"Failed to load gap store: {e}")
                self.gaps = []
