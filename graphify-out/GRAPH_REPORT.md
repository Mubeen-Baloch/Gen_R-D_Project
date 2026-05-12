# Graph Report - .  (2026-05-10)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 407 nodes · 668 edges · 30 communities (17 shown, 13 thin omitted)
- Extraction: 75% EXTRACTED · 25% INFERRED · 0% AMBIGUOUS · INFERRED: 166 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `3285c8ae`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]

## God Nodes (most connected - your core abstractions)
1. `DSKG` - 32 edges
2. `PipelineState` - 19 edges
3. `SynthesizerAgent` - 17 edges
4. `ContradictionDetectorAgent` - 14 edges
5. `GapAnalyzerAgent` - 13 edges
6. `RetrieverAgent` - 13 edges
7. `Paper` - 13 edges
8. `ClaimExtractorAgent` - 11 edges
9. `ConfidenceScorerAgent` - 11 edges
10. `TemporalTrackerAgent` - 11 edges

## Surprising Connections (you probably didn't know these)
- `run_study()` --calls--> `get_settings()`  [INFERRED]
  ablation_study.py → src/config/settings.py
- `main()` --calls--> `get_settings()`  [INFERRED]
  run.py → src/config/settings.py
- `run_study()` --calls--> `AblationAutonomyLoop`  [INFERRED]
  ablation_study.py → src/orchestration/ablation_loop.py
- `ClaimExtractorAgent` --uses--> `ClaimType`  [INFERRED]
  src/agents/claim_extractor.py → src/models/data_models.py
- `ClaimExtractorAgent` --uses--> `PipelineState`  [INFERRED]
  src/agents/claim_extractor.py → src/models/data_models.py

## Communities (30 total, 13 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (50): GapAnalyzerAgent, Gap Analyzer Agent — FRGO formalization (§4.4, §5.2.3). Uses DSKG structural an, Generate gaps directly from claim analysis when DSKG is insufficient., Formalize a structural gap indicator into FRGO gap objects using LLM., Resolution Agent — generates reconciliation statements for confirmed contradicti, Generate a full ConflictObject with reconciliation statement., Temporal Knowledge Evolution Tracker (TKET) — §4.6. Models diachronic trajector, Analyze a single research thread for temporal evolution. (+42 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (35): Extract claims from all processed papers in the corpus., Compute confidence scores for all claims in the pipeline state., Run two-stage contradiction detection on all claims., Evaluate the synthesized literature review., Construct the DSKG from the current state., Analyze the DSKG for structural gaps and formalize them as FRGO objects., Process the user topic and produce:           - A refined, expanded research qu, Generate a task execution plan based on the refined query and subtopics. (+27 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (26): Base agent class providing LLM client initialization, structured output parsing, Base class for all specialized agents in the framework.     Provides:       -, Initialize the LangChain LLM client based on provider config., Send a prompt to the LLM and return the raw response text., Send a prompt to the LLM and parse the response as JSON.         Handles markdo, Extract and parse JSON from LLM response.         Handles responses wrapped in, ConfidenceScorerAgent, Confidence Scorer — implements CGCERS formula (§4.2.2). Computes confidence sco (+18 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (24): Get summary statistics of the DSKG., Save the DSKG to a JSON file., Load the DSKG from a JSON file., Clear the entire graph., Clear all claims from the store., Conflict Registry — structured storage for contradiction objects (§4.3.3). Pers, Load from JSON if file exists., Structured database of all identified contradictions.     Directly integrated i (+16 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (19): DSKG, Dynamic Scientific Knowledge Graph (DSKG) — §4.5 NetworkX-based heterogeneous d, Add a finding/result node., Add a typed epistemic edge between two nodes., Claim A provides evidence for Claim B., Claim A conflicts with Claim B (bidirectional)., Paper A builds upon Paper B., Method is applied to a domain/concept. (+11 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (23): Retriever Agent — orchestrates Semantic Scholar + arXiv to build paper corpus (§, Execute targeted retrieval for specific queries (used by autonomy loop, Search Semantic Scholar with error handling., Search arXiv with error handling., RetrieverAgent, Paper, A research paper with metadata and parsed content., Get a specific section by type. (+15 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (17): ClaimExtractorAgent, Claim Extractor Agent — extracts atomic scientific claims from papers (§4.2.1, §, Extract atomic claims from a single paper., AtomicClaim, ConfidenceGradedClaim, An atomic scientific claim extracted from a paper (§4.2.1)., A claim with CGCERS confidence scoring (§4.2.2)., Compute the CGCERS confidence score. (+9 more)

### Community 7 - "Community 7"
Cohesion: 0.11
Nodes (15): A parsed section of a research paper., Section, PDF processing pipeline using GROBID (primary) with PyMuPDF fallback. Extracts, Process a batch of papers that have pdf_path set., Process PDF using GROBID's processFulltextDocument endpoint., Parse GROBID TEI XML output into a Paper object., Parse a single TEI div element into a Section., Extract all text content from an XML element, including children. (+7 more)

### Community 8 - "Community 8"
Cohesion: 0.14
Nodes (12): AutonomyLoop, Main execution script for the Ablation Study (§7.5). Runs the system across 6 va, run_study(), AblationAutonomyLoop, Ablation Autonomy Loop — allows disabling specific components for evaluation (§7, A modified AutonomyLoop that allows toggling core components for ablation studie, Execute one full pass, respecting ablation toggles., Autonomy Loop — orchestrates the multi-agent framework (§4.7). Manages state, i (+4 more)

### Community 9 - "Community 9"
Cohesion: 0.2
Nodes (9): _download_url(), Paper downloader — fetches open-access PDFs via direct URLs and Unpaywall. Mana, Try to find and download open-access PDF via Unpaywall., Download PDF from arXiv., Generate a safe filename for a paper PDF., Downloads open-access PDFs and manages a local cache., Attempt to download the PDF for a paper.         Tries: direct open-access URL, Download PDFs for a batch of papers.         Returns mapping of paper_id -> loc (+1 more)

### Community 10 - "Community 10"
Cohesion: 0.15
Nodes (10): BaseSettings, Config, get_settings(), Centralized configuration management using Pydantic Settings. Loads from .env f, Get or create the global settings instance., Application settings loaded from .env file., Create all required data directories., Return kwargs for initializing the LangChain LLM based on provider. (+2 more)

### Community 11 - "Community 11"
Cohesion: 0.19
Nodes (8): ContradictionDetectorAgent, Contradiction Detector Agent — two-stage CDRA (§4.3, §5.2.2). Stage 1: NLI mode, Compute NLI contradiction probability for a claim pair.         Uses cross-enco, Stage 2: LLM-based confirmation and classification.         For each candidate, Initialize the pre-trained NLI model for Stage 1 filtering., Stage 1: Identify candidate contradiction pairs using:         1. Embedding sim, AgentMessage, Structured message for inter-agent communication (§4.7.2).

### Community 12 - "Community 12"
Cohesion: 0.31
Nodes (9): compute_calib_score(), compute_ccs(), compute_cd_f1(), compute_gp_k(), compute_tcs(), get_all_metrics(), MetricsEvaluator, Evaluation Metrics (§5.4). Computes CCS, CD-F1, GP@K, TCS, and CalibScore based (+1 more)

### Community 13 - "Community 13"
Cohesion: 0.33
Nodes (5): Streamlit Frontend for Uncertainty-Aware Scientific Claim Synthesis. Provides a, Run the autonomy loop in a separate thread to prevent UI blocking., Render a styled metric card., render_metric_card(), run_pipeline()

## Knowledge Gaps
- **207 isolated node(s):** `Main execution script for the Ablation Study (§7.5). Runs the system across 6 va`, `Streamlit Frontend for Uncertainty-Aware Scientific Claim Synthesis. Provides a`, `Run the autonomy loop in a separate thread to prevent UI blocking.`, `Render a styled metric card.`, `CLI Entry Point for the Uncertainty-Aware Scientific Claim Synthesis system.` (+202 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DSKG` connect `Community 4` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 5`, `Community 6`, `Community 8`?**
  _High betweenness centrality (0.187) - this node is a cross-community bridge._
- **Why does `SynthesizerAgent` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 8`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Why does `PipelineState` connect `Community 0` to `Community 1`, `Community 2`, `Community 5`, `Community 6`, `Community 8`, `Community 11`, `Community 12`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `DSKG` (e.g. with `DSKGBuilderAgent` and `GapAnalyzerAgent`) actually correct?**
  _`DSKG` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `AutonomyLoop` (e.g. with `ClaimExtractorAgent` and `ConfidenceScorerAgent`) actually correct?**
  _`AutonomyLoop` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `BaseAgent` (e.g. with `ClaimExtractorAgent` and `ConfidenceScorerAgent`) actually correct?**
  _`BaseAgent` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `PipelineState` (e.g. with `ClaimExtractorAgent` and `ConfidenceScorerAgent`) actually correct?**
  _`PipelineState` has 16 INFERRED edges - model-reasoned connections that need verification._