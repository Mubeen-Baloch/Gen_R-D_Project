"""
Embedding generation wrapper.
Supports Google Gemini embeddings (primary) and sentence-transformers (local fallback).
"""

from __future__ import annotations

import numpy as np
from loguru import logger


class EmbeddingManager:
    """
    Manages embedding generation with provider abstraction.
    Primary: Google Gemini embedding API (requires API key).
    Fallback: sentence-transformers all-MiniLM-L6-v2 (local, free).
    """

    def __init__(self, provider: str = "google", model_name: str = "models/embedding-001", api_key: str = ""):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self._local_model = None
        self._genai = None

        if provider == "google" and api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._genai = genai
                logger.info(f"Initialized Google Gemini embeddings with model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to init Google embeddings: {e}. Falling back to local.")
                self.provider = "local"
        else:
            # Force local if provider is not google or api_key is missing/leaked
            self.provider = "local"
        
        if self.provider == "local":
            self._init_local_model()

    def _init_local_model(self):
        """Initialize local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Initialized local sentence-transformers embeddings")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers: {e}")
            raise

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string."""
        if self.provider == "google" and self._genai:
            try:
                result = self._genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document",
                )
                return result["embedding"]
            except Exception as e:
                logger.warning(f"Google embedding failed: {e}. Using local fallback.")
                if self._local_model is None:
                    self._init_local_model()
                return self._local_model.encode(text).tolist()
        else:
            return self._local_model.encode(text).tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if self.provider == "google" and self._genai:
            try:
                results = []
                # Gemini has batch limits, process in chunks
                batch_size = 100
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    for t in batch:
                        result = self._genai.embed_content(
                            model=self.model_name,
                            content=t,
                            task_type="retrieval_document",
                        )
                        results.append(result["embedding"])
                return results
            except Exception as e:
                logger.warning(f"Google batch embedding failed: {e}. Using local fallback.")
                if self._local_model is None:
                    self._init_local_model()
                return self._local_model.encode(texts).tolist()
        else:
            return self._local_model.encode(texts).tolist()

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
