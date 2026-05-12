"""
Base agent class providing LLM client initialization, structured output
parsing, logging, and message bus integration.
Supports Google Gemini (primary), OpenAI, and Anthropic providers.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.config.settings import get_settings


class BaseAgent:
    """
    Base class for all specialized agents in the framework.
    Provides:
      - LLM client initialization with provider abstraction
      - Structured JSON output parsing
      - Logging
      - Message bus integration
    """

    agent_name: str = "BaseAgent"
    agent_role: str = ""
    agent_goal: str = ""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.llm = self._init_llm()
        self._call_count = 0

    def _init_llm(self):
        """Initialize the LangChain LLM client based on provider config."""
        provider = self.settings.llm_provider

        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=self.settings.llm_model,
                google_api_key=self.settings.google_api_key,
                temperature=self.settings.temperature,
                max_output_tokens=self.settings.max_output_tokens,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.settings.llm_model,
                api_key=self.settings.openai_api_key,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_output_tokens,
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.settings.llm_model,
                api_key=self.settings.anthropic_api_key,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_output_tokens,
            )
        elif provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=self.settings.llm_model,
                groq_api_key=self.settings.groq_api_key,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_output_tokens,
            )
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=self.settings.llm_model,
                base_url=self.settings.ollama_base_url,
                temperature=self.settings.temperature,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def invoke(self, prompt: str, system_prompt: str = "") -> str:
        """
        Send a prompt to the LLM and return the raw response text.
        """
        self._call_count += 1
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"[{self.agent_name}] LLM invocation failed: {e}")
            raise

    def invoke_json(self, prompt: str, system_prompt: str = "") -> dict:
        """
        Send a prompt to the LLM and parse the response as JSON.
        Handles markdown code fences and partial JSON gracefully.
        """
        raw = self.invoke(prompt, system_prompt)
        return self._parse_json(raw)

    def _parse_json(self, text: str) -> dict:
        """
        Extract and parse JSON from LLM response.
        Handles responses wrapped in ```json ... ``` code fences.
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from code fences
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object/array boundaries
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            if start != -1:
                # Find matching closing bracket
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == start_char:
                        depth += 1
                    elif text[i] == end_char:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[start : i + 1])
                            except json.JSONDecodeError:
                                break

        logger.warning(f"[{self.agent_name}] Failed to parse JSON from response: {text[:200]}...")
        return {}

    def log_action(self, action: str, details: str = ""):
        """Log an agent action."""
        logger.info(f"[{self.agent_name}] {action}" + (f": {details}" if details else ""))
