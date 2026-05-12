"""
Goal Interpreter Agent — takes user topic and produces refined query + subtopic map.
First agent in the pipeline (§4.1 execution flow).
"""

from __future__ import annotations

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.data_models import PipelineState
from src.utils.prompts import GOAL_INTERPRETER_PROMPT


class GoalInterpreterAgent(BaseAgent):
    agent_name = "GoalInterpreter"
    agent_role = "Research topic decomposition and query refinement"
    agent_goal = "Transform a user-specified research topic into a precise query and comprehensive subtopic map"

    def run(self, state: PipelineState) -> PipelineState:
        """
        Process the user topic and produce:
          - A refined, expanded research query
          - A subtopic decomposition map with search queries
        """
        self.log_action("Interpreting research topic", state.topic)

        prompt = GOAL_INTERPRETER_PROMPT.format(topic=state.topic)
        system = (
            "You are a research librarian and domain expert. "
            "Your task is to decompose research topics into comprehensive, "
            "well-structured subtopic maps for systematic literature review."
        )

        result = self.invoke_json(prompt, system)

        if not result:
            logger.warning("Goal interpreter returned empty result; using topic as-is")
            state.refined_query = state.topic
            state.subtopics = [state.topic]
            return state

        state.refined_query = result.get("refined_query", state.topic)

        # Extract subtopics
        subtopics_data = result.get("subtopics", [])
        state.subtopics = []
        for st in subtopics_data:
            if isinstance(st, dict):
                name = st.get("name", "")
                if name:
                    state.subtopics.append(name)
            elif isinstance(st, str):
                state.subtopics.append(st)

        if not state.subtopics:
            state.subtopics = [state.topic]

        self.log_action(
            "Topic decomposed",
            f"Refined query: '{state.refined_query[:80]}...' | {len(state.subtopics)} subtopics"
        )

        # Store the full result for downstream use
        state.message_bus.append(
            __import__("src.models.data_models", fromlist=["AgentMessage"]).AgentMessage(
                sender=self.agent_name,
                recipient="PlannerAgent",
                payload=result,
            )
        )

        return state
