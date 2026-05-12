"""
Planner Agent — generates task execution DAG from subtopic map (§4.7.1).
"""

from __future__ import annotations

import json

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.data_models import PipelineState, AgentMessage
from src.utils.prompts import PLANNER_PROMPT


class PlannerAgent(BaseAgent):
    agent_name = "Planner"
    agent_role = "Task execution planning and scheduling"
    agent_goal = "Create an optimal task execution plan based on the subtopic decomposition"

    def run(self, state: PipelineState) -> PipelineState:
        """Generate a task execution plan based on the refined query and subtopics."""
        self.log_action("Creating task execution plan")

        # Get the goal interpreter's full output if available
        interpreter_msg = None
        for msg in state.message_bus:
            if msg.sender == "GoalInterpreter":
                interpreter_msg = msg
                break

        subtopics_str = json.dumps(state.subtopics, indent=2)

        prompt = PLANNER_PROMPT.format(
            refined_query=state.refined_query,
            subtopics=subtopics_str,
            max_papers=self.settings.max_papers,
        )

        system = (
            "You are a project manager specializing in research methodology. "
            "Create detailed, actionable task plans for systematic literature reviews."
        )

        result = self.invoke_json(prompt, system)

        if result:
            state.message_bus.append(
                AgentMessage(
                    sender=self.agent_name,
                    recipient="broadcast",
                    payload=result,
                )
            )
            self.log_action("Task plan created", f"{len(result.get('phases', []))} phases")
        else:
            # Create a default plan
            default_plan = self._create_default_plan(state)
            state.message_bus.append(
                AgentMessage(
                    sender=self.agent_name,
                    recipient="broadcast",
                    payload=default_plan,
                )
            )
            self.log_action("Using default task plan")

        return state

    def _create_default_plan(self, state: PipelineState) -> dict:
        """Create a sensible default plan when LLM planning fails."""
        papers_per_subtopic = max(5, self.settings.max_papers // max(1, len(state.subtopics)))

        return {
            "phases": [
                {
                    "phase_name": "Retrieval",
                    "description": "Retrieve papers for all subtopics",
                    "tasks": [
                        {
                            "task_id": f"T-{i:03d}",
                            "task_name": f"Retrieve papers for: {st}",
                            "agent": "RetrieverAgent",
                            "priority": 1,
                        }
                        for i, st in enumerate(state.subtopics)
                    ],
                },
                {
                    "phase_name": "Processing",
                    "description": "Process PDFs and extract claims",
                    "tasks": [
                        {"task_id": "T-100", "task_name": "Process PDFs", "agent": "PDFProcessor"},
                        {"task_id": "T-101", "task_name": "Extract claims", "agent": "ClaimExtractor"},
                        {"task_id": "T-102", "task_name": "Score confidence", "agent": "ConfidenceScorer"},
                    ],
                },
                {
                    "phase_name": "Analysis",
                    "description": "Build DSKG, detect contradictions, identify gaps",
                    "tasks": [
                        {"task_id": "T-200", "task_name": "Build DSKG", "agent": "DSKGBuilder"},
                        {"task_id": "T-201", "task_name": "Detect contradictions", "agent": "ContradictionDetector"},
                        {"task_id": "T-202", "task_name": "Analyze gaps", "agent": "GapAnalyzer"},
                        {"task_id": "T-203", "task_name": "Track temporal evolution", "agent": "TemporalTracker"},
                    ],
                },
                {
                    "phase_name": "Synthesis",
                    "description": "Generate and evaluate literature review",
                    "tasks": [
                        {"task_id": "T-300", "task_name": "Synthesize review", "agent": "Synthesizer"},
                        {"task_id": "T-301", "task_name": "Evaluate quality", "agent": "Critic"},
                    ],
                },
            ],
            "estimated_papers_per_subtopic": papers_per_subtopic,
        }
