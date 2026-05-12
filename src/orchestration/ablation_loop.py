"""
Ablation Autonomy Loop — allows disabling specific components for evaluation (§7.5).
"""

from __future__ import annotations
from src.orchestration.autonomy_loop import AutonomyLoop
from src.models.data_models import PipelineState
from loguru import logger

class AblationAutonomyLoop(AutonomyLoop):
    """
    A modified AutonomyLoop that allows toggling core components for ablation studies.
    """

    def __init__(self, settings=None, enabled_components: dict[str, bool] = None):
        super().__init__(settings)
        # Default all to True if not provided
        self.enabled = enabled_components or {
            "cgcers": True,     # Confidence Scoring
            "cdra": True,       # Contradiction Detection
            "frgo": True,       # Formalized Gaps
            "dskg": True,       # Knowledge Graph
            "tket": True,       # Temporal Tracking
        }

    def _execute_iteration(self, state: PipelineState) -> PipelineState:
        """Execute one full pass, respecting ablation toggles."""
        
        # 1. Retrieval (Always enabled for context)
        if not state.papers:
            state = self.retriever.run(state)

        # 2. PDF Processing (Always enabled)
        unprocessed = [p for p in state.papers.values() if not p.is_processed and p.pdf_path]
        if unprocessed:
            self.pdf_processor.process_batch(unprocessed)
            state.status = "processing"

        # 3. Extraction & Scoring
        if not state.claims or state.current_iteration > 1:
            state = self.extractor.run(state)
            
            if self.enabled.get("cgcers", True):
                state = self.scorer.run(state)
            else:
                logger.info("[ABLATION] Skipping Confidence Scoring (CGCERS)")

        state.status = "analyzing"

        # 4. Conflict Analysis
        if self.enabled.get("cdra", True):
            state = self.detector.run(state)
            state = self.resolver.run(state)
            
            # 5. Update confidence scores (only if scoring is also enabled)
            if self.enabled.get("cgcers", True):
                self.scorer.update_contradiction_scores(state)
        else:
            logger.info("[ABLATION] Skipping Contradiction Detection (CDRA)")

        # 6. DSKG Construction
        if self.enabled.get("dskg", True):
            state = self.dskg_builder.run(state)
        else:
            logger.info("[ABLATION] Skipping DSKG Construction")

        # 7. Advanced Analysis (Gaps & Temporal)
        if self.enabled.get("frgo", True):
            state = self.gap_analyzer.run(state)
        else:
            logger.info("[ABLATION] Skipping Gap Analysis (FRGO)")

        if self.enabled.get("tket", True):
            state = self.temporal_tracker.run(state)
        else:
            logger.info("[ABLATION] Skipping Temporal Tracking (TKET)")

        # 8. Synthesis & Evaluation (Always enabled to get scores)
        state = self.synthesizer.run(state)
        state = self.critic.run(state)

        return state
