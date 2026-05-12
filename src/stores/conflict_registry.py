"""
Conflict Registry — structured storage for contradiction objects (§4.3.3).
Persists to JSON and provides query/filter methods.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from src.models.data_models import ConflictObject, ConflictType, ConflictSeverity


class ConflictRegistry:
    """
    Structured database of all identified contradictions.
    Directly integrated into the literature review as a dedicated
    'Contested Claims and Methodological Disputes' section.
    """

    def __init__(self, persist_path: str = "./data/conflict_registry.json"):
        self.persist_path = persist_path
        self.conflicts: list[ConflictObject] = []
        self._load()

    def add_conflict(self, conflict: ConflictObject):
        """Add a confirmed conflict to the registry."""
        # Avoid duplicates
        existing_ids = {c.conflict_id for c in self.conflicts}
        if conflict.conflict_id not in existing_ids:
            self.conflicts.append(conflict)
            self._save()
            logger.debug(f"Added conflict {conflict.conflict_id}: {conflict.conflict_type.value}")

    def add_conflicts_batch(self, conflicts: list[ConflictObject]):
        """Add multiple conflicts at once."""
        existing_ids = {c.conflict_id for c in self.conflicts}
        added = 0
        for conflict in conflicts:
            if conflict.conflict_id not in existing_ids:
                self.conflicts.append(conflict)
                existing_ids.add(conflict.conflict_id)
                added += 1
        if added > 0:
            self._save()
            logger.info(f"Added {added} conflicts to registry (total: {len(self.conflicts)})")

    def get_by_type(self, conflict_type: ConflictType) -> list[ConflictObject]:
        """Get all conflicts of a specific type."""
        return [c for c in self.conflicts if c.conflict_type == conflict_type]

    def get_by_severity(self, severity: ConflictSeverity) -> list[ConflictObject]:
        """Get all conflicts of a specific severity."""
        return [c for c in self.conflicts if c.severity == severity]

    def get_by_paper(self, paper_id: str) -> list[ConflictObject]:
        """Get all conflicts involving a specific paper."""
        return [
            c for c in self.conflicts
            if c.source_paper_a_id == paper_id or c.source_paper_b_id == paper_id
        ]

    def get_by_claim(self, claim_id: str) -> list[ConflictObject]:
        """Get all conflicts involving a specific claim."""
        return [
            c for c in self.conflicts
            if c.pair.claim_a_id == claim_id or c.pair.claim_b_id == claim_id
        ]

    def get_stats(self) -> dict:
        """Get summary statistics of the conflict registry."""
        type_counts = {}
        severity_counts = {}
        for c in self.conflicts:
            type_counts[c.conflict_type.value] = type_counts.get(c.conflict_type.value, 0) + 1
            severity_counts[c.severity.value] = severity_counts.get(c.severity.value, 0) + 1

        return {
            "total": len(self.conflicts),
            "by_type": type_counts,
            "by_severity": severity_counts,
        }

    def to_narrative(self) -> str:
        """Generate a narrative summary of the conflict registry for the review."""
        if not self.conflicts:
            return "No significant contradictions were identified in the reviewed literature."

        sections = []
        for conflict_type in ConflictType:
            typed_conflicts = self.get_by_type(conflict_type)
            if not typed_conflicts:
                continue

            type_label = conflict_type.value.replace("_", " ").title()
            section = f"### {type_label} Conflicts ({len(typed_conflicts)} identified)\n\n"

            for conflict in typed_conflicts:
                section += f"**{conflict.conflict_id}**: "
                section += f"Claim from [{conflict.source_paper_a_id}] vs [{conflict.source_paper_b_id}]\n"
                section += f"- *Claim A*: {conflict.pair.claim_a_text[:200]}\n"
                section += f"- *Claim B*: {conflict.pair.claim_b_text[:200]}\n"
                section += f"- *Reconciliation*: {conflict.reconciliation_statement[:300]}\n\n"

            sections.append(section)

        return "\n".join(sections)

    def count(self) -> int:
        """Total number of conflicts in the registry."""
        return len(self.conflicts)

    def clear(self):
        """Clear all conflicts."""
        self.conflicts = []
        self._save()

    def _save(self):
        """Persist to JSON."""
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        data = [c.model_dump(mode="json") for c in self.conflicts]
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        """Load from JSON if file exists."""
        path = Path(self.persist_path)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.conflicts = [ConflictObject.model_validate(d) for d in data]
                logger.info(f"Loaded {len(self.conflicts)} conflicts from {self.persist_path}")
            except Exception as e:
                logger.warning(f"Failed to load conflict registry: {e}")
                self.conflicts = []
