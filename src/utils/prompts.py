"""
All structured LLM prompt templates used by the agents.
Each prompt is designed for structured JSON output to ensure reliable parsing.
"""

# ═══════════════════════════════════════════════════════════════════════
#  Goal Interpreter Agent
# ═══════════════════════════════════════════════════════════════════════

GOAL_INTERPRETER_PROMPT = """You are a scientific research expert. Given a research topic, produce a refined, precise query and decompose it into logical subtopics for a comprehensive literature review.

**Research Topic:** {topic}

Return your response as a JSON object with this exact schema:
{{
  "refined_query": "A precise, expanded version of the topic suitable for academic search",
  "subtopics": [
    {{
      "name": "Subtopic name",
      "description": "Brief description of what this subtopic covers",
      "search_queries": ["query1", "query2"]
    }}
  ],
  "domain": "Primary research domain",
  "key_terms": ["term1", "term2", "term3"],
  "temporal_focus": "Suggested year range for the review"
}}

Generate 5-8 subtopics that together provide comprehensive coverage of the research area. Each subtopic should have 2-3 specific search queries for academic databases."""

# ═══════════════════════════════════════════════════════════════════════
#  Planner Agent
# ═══════════════════════════════════════════════════════════════════════

PLANNER_PROMPT = """You are a research project planner. Given a refined research query and its subtopics, create a task execution plan for generating a comprehensive literature review.

**Refined Query:** {refined_query}
**Subtopics:** {subtopics}
**Max Papers:** {max_papers}

Return your response as a JSON object with this exact schema:
{{
  "phases": [
    {{
      "phase_name": "Phase name",
      "description": "What this phase accomplishes",
      "tasks": [
        {{
          "task_id": "T-001",
          "task_name": "Task name",
          "description": "Task description",
          "depends_on": [],
          "agent": "agent_name",
          "priority": 1
        }}
      ]
    }}
  ],
  "estimated_papers_per_subtopic": 10,
  "retrieval_strategy": "Description of how to distribute retrieval across subtopics"
}}"""

# ═══════════════════════════════════════════════════════════════════════
#  Claim Extractor Agent (§5.2.1)
# ═══════════════════════════════════════════════════════════════════════

CLAIM_EXTRACTION_PROMPT = """You are a scientific claim extraction expert. Extract all atomic scientific claims from the following research paper section. An atomic claim is a single, indivisible assertion that can be independently evaluated.

**Paper Title:** {paper_title}
**Paper Year:** {paper_year}
**Section Type:** {section_type}
**Section Content:**
{section_content}

For each claim, specify:
- claim_text: The precise propositional assertion
- claim_type: One of "method" (technique/approach proposed or used), "result" (empirical outcomes, benchmarks), or "theoretical" (explanations, hypotheses, positions)
- confidence_indicators: Any hedging language detected (e.g., "suggests", "demonstrates", "proves", "may", "appears to")
- subject_entities: Key entities the claim is about (models, datasets, methods, concepts)
- condition_qualifiers: Any conditions or limitations mentioned

Return a JSON object:
{{
  "claims": [
    {{
      "claim_text": "...",
      "claim_type": "method|result|theoretical",
      "confidence_indicators": ["..."],
      "subject_entities": ["..."],
      "condition_qualifiers": ["..."]
    }}
  ]
}}

Extract 3-15 claims per section. Be precise and atomic — each claim should express exactly one assertion."""

# ═══════════════════════════════════════════════════════════════════════
#  Contradiction Detection Agent (§5.2.2)
# ═══════════════════════════════════════════════════════════════════════

CONTRADICTION_CONFIRMATION_PROMPT = """You are a scientific reasoning expert specializing in identifying and classifying contradictions between research claims.

Given these two scientific claims from different papers, determine:

**Claim A** (from paper {paper_a_title}, {paper_a_year}):
"{claim_a_text}"

**Claim B** (from paper {paper_b_title}, {paper_b_year}):
"{claim_b_text}"

Analyze the relationship and return a JSON object:
{{
  "severity": "contradictory|partially_contradictory|non_contradictory",
  "conflict_type": "methodological|domain_specificity|temporal|definitional",
  "explanation": "Detailed explanation of why these claims conflict or don't conflict",
  "reconciliation_statement": "If contradictory: a paragraph explaining how both claims can be understood within a unified framework, specifying conditions under which each holds. If non-contradictory: explain why they are compatible."
}}

**Conflict Type Definitions:**
- methodological: Different conclusions due to different methods, metrics, or experimental setups
- domain_specificity: Claims hold in different application domains
- temporal: One claim superseded by more recent findings
- definitional: Claims use different definitions for the same term"""

# ═══════════════════════════════════════════════════════════════════════
#  Gap Analyzer Agent (§5.2.3)
# ═══════════════════════════════════════════════════════════════════════

GAP_ANALYSIS_PROMPT = """You are a research gap identification expert. Based on the following structural evidence from a scientific knowledge graph, formulate research gaps as structured objects.

**Topic Cluster:** {topic_cluster}
**Structural Evidence:**
{structural_evidence}

**Existing Claims in this Cluster:**
{cluster_claims}

**Cross-Cluster Connections:**
{cross_cluster_info}

For each identified gap, produce a structured gap object following this schema:
{{
  "gaps": [
    {{
      "gap_type": "unexplored_intersection|underexplored_area|contradictory_state|methodological",
      "topic_cluster": "...",
      "missing_intersection": ["topic1", "topic2"],
      "gap_statement": "A specific, precise statement of what is missing in the literature",
      "evidence_papers": ["paper_id1", "paper_id2"],
      "evidence_type": "implicit_boundary|explicit_mention|structural_absence",
      "implied_by": ["Evidence description 1", "Evidence description 2"],
      "confidence": 0.0,
      "temporal_position": "persistent_since_YYYY|recent_emergence|declining_relevance",
      "gap_class": "methodological|theoretical|empirical|application",
      "falsifiability": "A description of what a paper filling this gap would contain"
    }}
  ]
}}

Generate 2-5 specific, evidence-grounded gaps. Each gap must have a falsifiability statement."""

# ═══════════════════════════════════════════════════════════════════════
#  Temporal Tracker Agent (§4.6)
# ═══════════════════════════════════════════════════════════════════════

TEMPORAL_ANALYSIS_PROMPT = """You are an expert in tracking the temporal evolution of scientific research threads.

**Research Thread Topic:** {thread_topic}
**Claims ordered chronologically:**
{chronological_claims}

**Method appearances over time:**
{method_timeline}

Analyze the temporal evolution and return a JSON object:
{{
  "trajectory_type": "growing_consensus|diverging|reversal|stable",
  "narrative": "A diachronic narrative describing how understanding of this topic has evolved over time. Use the format: 'Early work from YYYY-YYYY established X. This was qualified in YYYY-YYYY by evidence that... The current consensus as of YYYY is...'",
  "consensus_shifts": [
    {{
      "year_range": "YYYY-YYYY",
      "description": "What shifted in the consensus",
      "key_papers": ["paper_id1"]
    }}
  ],
  "method_emergence": {{
    "method_name": "YYYY (first appearance)"
  }},
  "claim_qualifications": [
    "Description of how an earlier claim was qualified or refined"
  ]
}}"""

# ═══════════════════════════════════════════════════════════════════════
#  Synthesizer Agent (§4.5.3)
# ═══════════════════════════════════════════════════════════════════════

SYNTHESIS_PROMPT = """You are a scientific literature review synthesizer. Generate a comprehensive, structured literature review section for the following thematic cluster.

**Section Theme:** {theme}
**Key Claims (with confidence levels):**
{claims_with_confidence}

**Key Contradictions in this Theme:**
{contradictions}

**Research Gaps Related to this Theme:**
{gaps}

**Temporal Evolution:**
{temporal_narrative}

**Guidelines:**
1. Integrate multiple papers — do NOT merely summarize individual papers
2. Mark claim confidence inline using [HIGH], [MODERATE], [CONTESTED], or [DISPUTED] tags
3. Acknowledge and discuss contradictions explicitly
4. Reference temporal evolution where relevant
5. Cite papers using their paper IDs in brackets, e.g., [paper_id]
6. Write in an authoritative academic tone appropriate for a survey paper

Return a JSON object:
{{
  "section_title": "A descriptive title for this review section",
  "content": "The full section text with inline confidence annotations and citations",
  "key_findings": ["Finding 1", "Finding 2"],
  "open_questions": ["Question 1"]
}}"""

# ═══════════════════════════════════════════════════════════════════════
#  Critic Agent (§4.7.3)
# ═══════════════════════════════════════════════════════════════════════

CRITIC_PROMPT = """You are a critical evaluator of automated literature reviews. Evaluate the following generated review against quality criteria.

**Original Topic:** {topic}
**Number of Papers Analyzed:** {num_papers}
**Generated Review:**
{review_text}

**Claims Extracted:** {num_claims}
**Contradictions Found:** {num_contradictions}
**Gaps Identified:** {num_gaps}

Evaluate on these dimensions (score each 0.0 to 1.0):

1. **Claim Coverage**: How comprehensively does the review cover the key claims in the field?
2. **Contradiction Completeness**: Are conflicting findings adequately identified and discussed?
3. **Gap Formalization Quality**: Are research gaps specific, evidence-grounded, and falsifiable?
4. **Temporal Coherence**: Does the review accurately convey the historical development of the field?
5. **Synthesis Coherence**: Does the review integrate papers into a coherent narrative rather than listing them?

Return a JSON object:
{{
  "claim_coverage_score": 0.0,
  "contradiction_completeness": 0.0,
  "gap_formalization_quality": 0.0,
  "temporal_coherence_score": 0.0,
  "synthesis_coherence_score": 0.0,
  "feedback": "Overall assessment and specific areas for improvement",
  "improvement_directives": [
    "Specific directive 1 for improvement",
    "Specific directive 2 for improvement"
  ]
}}"""

# ═══════════════════════════════════════════════════════════════════════
#  Resolution Agent
# ═══════════════════════════════════════════════════════════════════════

RESOLUTION_PROMPT = """You are a scientific conflict resolution expert. Given a confirmed contradiction between two research claims, generate a detailed reconciliation statement.

**Claim A** (from {paper_a_title}, {paper_a_year}):
"{claim_a_text}"

**Claim B** (from {paper_b_title}, {paper_b_year}):
"{claim_b_text}"

**Conflict Type:** {conflict_type}
**Initial Explanation:** {explanation}

Generate a comprehensive reconciliation statement that:
1. Presents both claims fairly
2. Explains the source of disagreement
3. Identifies conditions under which each claim holds
4. Synthesizes a qualified assertion that is true under specified conditions

Return a JSON object:
{{
  "reconciliation_statement": "A paragraph-length reconciliation...",
  "qualified_assertion": "A single qualified statement that reconciles both claims",
  "conditions_for_claim_a": "Under what conditions Claim A holds",
  "conditions_for_claim_b": "Under what conditions Claim B holds"
}}"""
