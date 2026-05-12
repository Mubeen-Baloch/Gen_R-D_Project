"""
Centralized configuration management using Pydantic Settings.
Loads from .env file and provides typed access to all configuration parameters.
"""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # ── LLM API Keys ──────────────────────────────────────────────
    google_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # ── Academic API Keys ─────────────────────────────────────────
    semantic_scholar_api_key: str = ""

    # ── LLM Configuration ─────────────────────────────────────────
    llm_provider: Literal["google", "openai", "anthropic", "groq", "ollama"] = "google"
    llm_model: str = "gemma-3-12b-it"
    embedding_model: str = "models/gemini-embedding-001"
    temperature: float = 0.1
    max_output_tokens: int = 8192

    # ── GROBID Configuration ──────────────────────────────────────
    grobid_server_url: str = "http://localhost:8070"
    use_grobid: bool = True  # Falls back to PyMuPDF if GROBID unavailable

    # ── Pipeline Configuration ────────────────────────────────────
    max_papers: int = 50
    max_autonomy_iterations: int = 3
    quality_threshold: float = 0.7
    temporal_scope_start: int = 2017
    temporal_scope_end: int = 2025

    # ── Confidence Scoring Weights (CGCERS) ───────────────────────
    alpha_consensus: float = 0.4
    beta_recency: float = 0.35
    gamma_contradiction: float = 0.25
    recency_decay: float = 0.15  # Lambda in the recency formula

    # ── Contradiction Detection Thresholds ────────────────────────
    embedding_similarity_threshold: float = 0.75
    contradiction_score_threshold: float = 0.6

    # ── Paths ─────────────────────────────────────────────────────
    data_dir: str = "./data"
    papers_dir: str = "./data/papers"
    processed_dir: str = "./data/processed"
    output_dir: str = "./data/output"
    dskg_path: str = "./data/dskg.json"
    conflict_registry_path: str = "./data/conflict_registry.json"
    gap_store_path: str = "./data/gap_store.json"
    claim_store_path: str = "./data/claim_store"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_directories(self):
        """Create all required data directories."""
        for dir_path in [self.data_dir, self.papers_dir, self.processed_dir, self.output_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def get_llm_kwargs(self) -> dict:
        """Return kwargs for initializing the LangChain LLM based on provider."""
        if self.llm_provider == "google":
            return {
                "model": self.llm_model,
                "google_api_key": self.google_api_key,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            }
        elif self.llm_provider == "openai":
            return {
                "model": self.llm_model,
                "api_key": self.openai_api_key,
                "temperature": self.temperature,
                "max_tokens": self.max_output_tokens,
            }
        elif self.llm_provider == "anthropic":
            return {
                "model": self.llm_model,
                "api_key": self.anthropic_api_key,
                "temperature": self.temperature,
                "max_tokens": self.max_output_tokens,
            }
        elif self.llm_provider == "groq":
            return {
                "model": self.llm_model,
                "groq_api_key": self.groq_api_key,
                "temperature": self.temperature,
                "max_tokens": self.max_output_tokens,
            }
        elif self.llm_provider == "ollama":
            return {
                "model": self.llm_model,
                "base_url": self.ollama_base_url,
                "temperature": self.temperature,
            }
        raise ValueError(f"Unknown LLM provider: {self.llm_provider}")


# Global singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings
