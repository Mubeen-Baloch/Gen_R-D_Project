"""
Pydantic data models for all core entities in the framework.
Covers Papers, Claims, Conflicts, Gaps, Temporal Threads, and Reviews.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ════════════════════════════════════════════════════════════════════
#  Enums
# ════════════════════════════════════════════════════════════════════

class ClaimType(str, Enum):
    """Three-level claim taxonomy from §4.2.1."""
    METHOD = "method"
    RESULT = "result"
    THEORETICAL = "theoretical"


class ConfidenceCategory(str, Enum):
    """Confidence categories from §4.2.2."""
    HIGH = "high"           # score >= 0.75
    MODERATE = "moderate"   # 0.5 <= score < 0.75
    CONTESTED = "contested" # 0.25 <= score < 0.5
    DISPUTED = "disputed"   # score < 0.25


class ConflictType(str, Enum):
    """Four-dimensional conflict taxonomy from §4.3.2."""
    METHODOLOGICAL = "methodological"
    DOMAIN_SPECIFICITY = "domain_specificity"
    TEMPORAL = "temporal"
    DEFINITIONAL = "definitional"


class ConflictSeverity(str, Enum):
    """Severity of a detected conflict."""
    CONTRADICTORY = "contradictory"
    PARTIALLY_CONTRADICTORY = "partially_contradictory"
    NON_CONTRADICTORY = "non_contradictory"


class GapType(str, Enum):
    """Gap type taxonomy from §4.4.2."""
    UNEXPLORED_INTERSECTION = "unexplored_intersection"
    UNDEREXPLORED_AREA = "underexplored_area"
    CONTRADICTORY_STATE = "contradictory_state"
    METHODOLOGICAL = "methodological"


class GapClass(str, Enum):
    """High-level gap classification."""
    METHODOLOGICAL = "methodological"
    THEORETICAL = "theoretical"
    EMPIRICAL = "empirical"
    APPLICATION = "application"


class DSKGNodeType(str, Enum):
    """Node types in the Dynamic Scientific Knowledge Graph (§4.5.1)."""
    CONCEPT = "concept"
    CLAIM = "claim"
    PAPER = "paper"
    METHOD = "method"
    FINDING = "finding"


class DSKGEdgeType(str, Enum):
    """Epistemic edge types in the DSKG (§4.5.1)."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    QUALIFIES = "qualifies"
    APPLIES_TO = "applies_to"
    EVALUATED_ON = "evaluated_on"
    INTRODUCES = "introduces"
    OUTPERFORMS = "outperforms"


class AgentMessageType(str, Enum):
    """Message types for inter-agent communication (§4.7.2)."""
    TASK_ASSIGNMENT = "task_assignment"
    RESULT_DELIVERY = "result_delivery"
    FEEDBACK = "feedback"
    QUERY = "query"


# ════════════════════════════════════════════════════════════════════
#  Paper & Section Models
# ════════════════════════════════════════════════════════════════════

class Section(BaseModel):
    """A parsed section of a research paper."""
    title: str = ""
    content: str = ""
    section_type: str = ""  # abstract, introduction, methods, results, discussion, conclusion
    tables: list[str] = Field(default_factory=list)
    figures: list[str] = Field(default_factory=list)


class Paper(BaseModel):
    """A research paper with metadata and parsed content."""
    paper_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int = 0
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    abstract: str = ""
    url: str = ""
    pdf_path: str = ""
    citation_count: int = 0
    references: list[str] = Field(default_factory=list)  # Paper IDs
    sections: list[Section] = Field(default_factory=list)
    full_text: str = ""
    is_processed: bool = False
    retrieved_at: datetime = Field(default_factory=datetime.now)

    def get_section(self, section_type: str) -> Optional[Section]:
        """Get a specific section by type."""
        for s in self.sections:
            if s.section_type.lower() == section_type.lower():
                return s
        return None


# ════════════════════════════════════════════════════════════════════
#  Claim Models (CGCERS)
# ════════════════════════════════════════════════════════════════════

class AtomicClaim(BaseModel):
    """An atomic scientific claim extracted from a paper (§4.2.1)."""
    claim_id: str = Field(default_factory=lambda: f"C-{uuid.uuid4().hex[:6]}")
    claim_text: str
    claim_type: ClaimType
    source_paper_id: str
    source_section: str = ""
    confidence_indicators: list[str] = Field(default_factory=list)  # hedging language
    subject_entities: list[str] = Field(default_factory=list)
    condition_qualifiers: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list, repr=False)


class ConfidenceGradedClaim(BaseModel):
    """A claim with CGCERS confidence scoring (§4.2.2)."""
    claim: AtomicClaim
    consensus_score: float = 0.0         # Proportion of supporting papers
    recency_score: float = 0.0           # Time-decay weighted support
    contradiction_score: float = 0.0     # Proportion of conflicting papers
    overall_confidence: float = 0.0      # Final composite score
    confidence_category: ConfidenceCategory = ConfidenceCategory.MODERATE
    supporting_paper_ids: list[str] = Field(default_factory=list)
    contradicting_paper_ids: list[str] = Field(default_factory=list)

    def compute_confidence(self, alpha: float = 0.4, beta: float = 0.35, gamma: float = 0.25):
        """Compute the CGCERS confidence score."""
        self.overall_confidence = (
            alpha * self.consensus_score
            + beta * self.recency_score
            - gamma * self.contradiction_score
        )
        # Clamp to [0, 1]
        self.overall_confidence = max(0.0, min(1.0, self.overall_confidence))
        # Assign category
        if self.overall_confidence >= 0.75:
            self.confidence_category = ConfidenceCategory.HIGH
        elif self.overall_confidence >= 0.5:
            self.confidence_category = ConfidenceCategory.MODERATE
        elif self.overall_confidence >= 0.25:
            self.confidence_category = ConfidenceCategory.CONTESTED
        else:
            self.confidence_category = ConfidenceCategory.DISPUTED


# ════════════════════════════════════════════════════════════════════
#  Conflict Models (CDRA)
# ════════════════════════════════════════════════════════════════════

class ConflictPair(BaseModel):
    """A pair of potentially conflicting claims (§4.3.1)."""
    pair_id: str = Field(default_factory=lambda: f"CP-{uuid.uuid4().hex[:6]}")
    claim_a_id: str
    claim_b_id: str
    claim_a_text: str = ""
    claim_b_text: str = ""
    embedding_similarity: float = 0.0
    nli_contradiction_score: float = 0.0  # From Stage 1 classifier


class ConflictObject(BaseModel):
    """A confirmed, classified conflict with reconciliation (§4.3.3)."""
    conflict_id: str = Field(default_factory=lambda: f"CF-{uuid.uuid4().hex[:6]}")
    pair: ConflictPair
    severity: ConflictSeverity = ConflictSeverity.CONTRADICTORY
    conflict_type: ConflictType = ConflictType.METHODOLOGICAL
    explanation: str = ""
    reconciliation_statement: str = ""
    source_paper_a_id: str = ""
    source_paper_b_id: str = ""
    detected_at: datetime = Field(default_factory=datetime.now)


# ════════════════════════════════════════════════════════════════════
#  Gap Models (FRGO)
# ════════════════════════════════════════════════════════════════════

class GapObject(BaseModel):
    """A formalized research gap object (§4.4.1)."""
    gap_id: str = Field(default_factory=lambda: f"G-{uuid.uuid4().hex[:4]}")
    gap_type: GapType = GapType.UNEXPLORED_INTERSECTION
    topic_cluster: str = ""
    missing_intersection: list[str] = Field(default_factory=list)
    gap_statement: str = ""
    evidence_papers: list[str] = Field(default_factory=list)  # Paper IDs
    evidence_type: str = "implicit_boundary"
    implied_by: list[str] = Field(default_factory=list)  # Evidence descriptions
    confidence: float = 0.0
    temporal_position: str = ""
    gap_class: GapClass = GapClass.METHODOLOGICAL
    falsifiability: str = ""  # What a gap-filling paper would contain
    detected_at: datetime = Field(default_factory=datetime.now)


# ════════════════════════════════════════════════════════════════════
#  Temporal Models (TKET)
# ════════════════════════════════════════════════════════════════════

class TemporalDataPoint(BaseModel):
    """A single data point in a temporal trajectory."""
    year: int
    claim_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    dominant_claim: str = ""
    consensus_strength: float = 0.0


class TemporalThread(BaseModel):
    """A research thread tracked over time (§4.6.1)."""
    thread_id: str = Field(default_factory=lambda: f"T-{uuid.uuid4().hex[:4]}")
    topic: str = ""
    trajectory_type: str = ""  # growing_consensus, diverging, reversal
    data_points: list[TemporalDataPoint] = Field(default_factory=list)
    method_emergence: dict[str, int] = Field(default_factory=dict)  # method -> first_year
    claim_qualifications: list[str] = Field(default_factory=list)
    narrative: str = ""  # Generated diachronic narrative


# ════════════════════════════════════════════════════════════════════
#  Review & Evaluation Models
# ════════════════════════════════════════════════════════════════════

class ReviewSection(BaseModel):
    """A section of the generated literature review."""
    title: str
    content: str
    theme_cluster: str = ""
    claim_ids: list[str] = Field(default_factory=list)
    confidence_annotations: dict[str, str] = Field(default_factory=dict)  # claim_id -> category


class GeneratedReview(BaseModel):
    """The final structured literature review output."""
    topic: str = ""
    generated_at: datetime = Field(default_factory=datetime.now)
    sections: list[ReviewSection] = Field(default_factory=list)
    contested_claims_section: str = ""  # Conflict Registry narrative
    research_gaps_section: str = ""     # FRGO narrative
    temporal_narrative: str = ""        # TKET narrative
    conclusion: str = ""
    total_papers_analyzed: int = 0
    total_claims_extracted: int = 0
    total_contradictions_found: int = 0
    total_gaps_identified: int = 0


class CriticEvaluation(BaseModel):
    """Quality assessment from the Critic Agent (§4.7.3)."""
    iteration: int = 0
    claim_coverage_score: float = 0.0
    contradiction_completeness: float = 0.0
    gap_formalization_quality: float = 0.0
    temporal_coherence_score: float = 0.0
    synthesis_coherence_score: float = 0.0
    overall_quality: float = 0.0
    feedback: str = ""
    improvement_directives: list[str] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=datetime.now)

    def compute_overall(self, weights: dict[str, float] | None = None):
        """Compute weighted average overall quality."""
        if weights is None:
            weights = {
                "claim_coverage": 0.25,
                "contradiction": 0.2,
                "gap": 0.2,
                "temporal": 0.15,
                "synthesis": 0.2,
            }
        self.overall_quality = (
            weights["claim_coverage"] * self.claim_coverage_score
            + weights["contradiction"] * self.contradiction_completeness
            + weights["gap"] * self.gap_formalization_quality
            + weights["temporal"] * self.temporal_coherence_score
            + weights["synthesis"] * self.synthesis_coherence_score
        )


# ════════════════════════════════════════════════════════════════════
#  Agent Communication Models
# ════════════════════════════════════════════════════════════════════

class AgentMessage(BaseModel):
    """Structured message for inter-agent communication (§4.7.2)."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender: str
    recipient: str = "broadcast"
    message_type: AgentMessageType = AgentMessageType.RESULT_DELIVERY
    payload: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


# ════════════════════════════════════════════════════════════════════
#  Pipeline State (shared across all agents)
# ════════════════════════════════════════════════════════════════════

class PipelineState(BaseModel):
    """
    Central shared state for the entire pipeline.
    All agents read from and write to this state.
    """
    # Input
    topic: str = ""
    refined_query: str = ""
    subtopics: list[str] = Field(default_factory=list)

    # Paper corpus
    papers: dict[str, Paper] = Field(default_factory=dict)  # paper_id -> Paper

    # Claims
    claims: dict[str, ConfidenceGradedClaim] = Field(default_factory=dict)  # claim_id -> Claim

    # Conflicts
    conflict_pairs: list[ConflictPair] = Field(default_factory=list)
    conflicts: list[ConflictObject] = Field(default_factory=list)

    # Gaps
    gaps: list[GapObject] = Field(default_factory=list)

    # Temporal
    temporal_threads: list[TemporalThread] = Field(default_factory=list)

    # Review
    review: Optional[GeneratedReview] = None

    # Evaluation
    evaluations: list[CriticEvaluation] = Field(default_factory=list)

    # Agent messages
    message_bus: list[AgentMessage] = Field(default_factory=list)

    # Status tracking
    current_iteration: int = 0
    status: str = "initialized"  # initialized, retrieving, processing, analyzing, synthesizing, complete
    errors: list[str] = Field(default_factory=list)
