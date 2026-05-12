"""
Claim Store — ChromaDB-backed vector store for claims.
Supports semantic similarity search for consensus computation and
contradiction candidate identification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
from loguru import logger

from src.models.data_models import AtomicClaim, ConfidenceGradedClaim


class ClaimStore:
    """
    ChromaDB-backed store for scientific claims with vector similarity search.
    Used by:
      - Confidence Scorer: find semantically equivalent claims for consensus
      - Contradiction Detector: find candidate claim pairs
      - Synthesizer: retrieve claims by topic
    """

    def __init__(self, persist_dir: str = "./data/claim_store"):
        self.persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="scientific_claims",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ClaimStore initialized at {persist_dir} ({self.collection.count()} claims)")

    def add_claim(self, claim: ConfidenceGradedClaim, embedding: list[float]):
        """Add a single claim with its embedding to the store."""
        self.collection.upsert(
            ids=[claim.claim.claim_id],
            embeddings=[embedding],
            documents=[claim.claim.claim_text],
            metadatas=[{
                "claim_type": claim.claim.claim_type.value,
                "source_paper_id": claim.claim.source_paper_id,
                "confidence": claim.overall_confidence,
                "confidence_category": claim.confidence_category.value,
                "subject_entities": ",".join(claim.claim.subject_entities),
            }],
        )

    def add_claims_batch(
        self,
        claims: list[ConfidenceGradedClaim],
        embeddings: list[list[float]],
    ):
        """Add multiple claims at once."""
        if not claims:
            return

        ids = [c.claim.claim_id for c in claims]
        documents = [c.claim.claim_text for c in claims]
        metadatas = [
            {
                "claim_type": c.claim.claim_type.value,
                "source_paper_id": c.claim.source_paper_id,
                "confidence": c.overall_confidence,
                "confidence_category": c.confidence_category.value,
                "subject_entities": ",".join(c.claim.subject_entities),
            }
            for c in claims
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(claims)} claims to store (total: {self.collection.count()})")

    def find_similar_claims(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        threshold: float = 0.75,
        exclude_paper_id: str = "",
    ) -> list[dict]:
        """
        Find semantically similar claims using vector search.
        Returns claims with similarity >= threshold.
        """
        where = None
        if exclude_paper_id:
            where = {"source_paper_id": {"$ne": exclude_paper_id}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        similar = []
        if results["ids"] and results["ids"][0]:
            for i, claim_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity for cosine
                distance = results["distances"][0][i] if results["distances"] else 1.0
                similarity = 1.0 - distance  # cosine distance -> similarity

                if similarity >= threshold:
                    similar.append({
                        "claim_id": claim_id,
                        "claim_text": results["documents"][0][i],
                        "similarity": similarity,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    })

        return similar

    def find_contradiction_candidates(
        self,
        claim_embedding: list[float],
        claim_paper_id: str,
        n_results: int = 20,
        similarity_threshold: float = 0.5,
    ) -> list[dict]:
        """
        Find potential contradiction candidates.
        These are claims from different papers with high semantic similarity
        (similar topic, potentially conflicting conclusions).
        """
        results = self.collection.query(
            query_embeddings=[claim_embedding],
            n_results=n_results,
            where={"source_paper_id": {"$ne": claim_paper_id}} if claim_paper_id else None,
            include=["documents", "metadatas", "distances"],
        )

        candidates = []
        if results["ids"] and results["ids"][0]:
            for i, claim_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                similarity = 1.0 - distance

                if similarity >= similarity_threshold:
                    candidates.append({
                        "claim_id": claim_id,
                        "claim_text": results["documents"][0][i],
                        "similarity": similarity,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    })

        return candidates

    def get_all_claims(self) -> list[dict]:
        """Retrieve all claims from the store."""
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.get(
            include=["documents", "metadatas"],
        )

        claims = []
        for i, claim_id in enumerate(results["ids"]):
            claims.append({
                "claim_id": claim_id,
                "claim_text": results["documents"][i],
                "metadata": results["metadatas"][i],
            })

        return claims

    def count(self) -> int:
        """Get the total number of claims in the store."""
        return self.collection.count()

    def clear(self):
        """Clear all claims from the store."""
        self.client.delete_collection("scientific_claims")
        self.collection = self.client.get_or_create_collection(
            name="scientific_claims",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ClaimStore cleared")
