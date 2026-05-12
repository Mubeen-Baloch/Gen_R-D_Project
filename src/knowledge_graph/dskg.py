"""
Dynamic Scientific Knowledge Graph (DSKG) — §4.5
NetworkX-based heterogeneous directed graph with typed nodes and epistemic edges.
Supports JSON persistence, subgraph extraction, and incremental construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx
from loguru import logger

from src.models.data_models import (
    AtomicClaim,
    ConfidenceGradedClaim,
    DSKGEdgeType,
    DSKGNodeType,
    Paper,
)


class DSKG:
    """
    Dynamic Scientific Knowledge Graph.
    
    A directed heterogeneous graph with 5 node types (Concept, Claim, Paper,
    Method, Finding) and 8 epistemic edge types (supports, contradicts,
    extends, qualifies, applies_to, evaluated_on, introduces, outperforms).
    
    Built on NetworkX with JSON persistence.
    """

    def __init__(self, persist_path: str = "./data/dskg.json"):
        self.graph = nx.DiGraph()
        self.persist_path = persist_path
        self._node_counter = {"concept": 0, "method": 0, "finding": 0}

    # ─── Node Operations ────────────────────────────────────────────

    def add_paper_node(self, paper: Paper) -> str:
        """Add a paper node to the graph."""
        node_id = f"paper:{paper.paper_id}"
        self.graph.add_node(
            node_id,
            node_type=DSKGNodeType.PAPER.value,
            title=paper.title,
            year=paper.year,
            authors=paper.authors[:5],  # Limit for storage
            venue=paper.venue,
            citation_count=paper.citation_count,
            abstract=paper.abstract[:500],
        )
        return node_id

    def add_claim_node(self, claim: ConfidenceGradedClaim) -> str:
        """Add a claim node with confidence metadata."""
        node_id = f"claim:{claim.claim.claim_id}"
        self.graph.add_node(
            node_id,
            node_type=DSKGNodeType.CLAIM.value,
            claim_text=claim.claim.claim_text,
            claim_type=claim.claim.claim_type.value,
            source_paper_id=claim.claim.source_paper_id,
            confidence=claim.overall_confidence,
            confidence_category=claim.confidence_category.value,
            subject_entities=claim.claim.subject_entities,
        )
        # Link claim to its source paper
        paper_node = f"paper:{claim.claim.source_paper_id}"
        if self.graph.has_node(paper_node):
            self.graph.add_edge(
                paper_node,
                node_id,
                edge_type=DSKGEdgeType.INTRODUCES.value,
                weight=1.0,
            )
        return node_id

    def add_concept_node(self, concept_name: str, description: str = "") -> str:
        """Add a concept node (research concept, dataset, metric, domain)."""
        node_id = f"concept:{concept_name.lower().replace(' ', '_')}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(
                node_id,
                node_type=DSKGNodeType.CONCEPT.value,
                name=concept_name,
                description=description,
            )
        return node_id

    def add_method_node(self, method_name: str, description: str = "", year: int = 0) -> str:
        """Add a method/technique node."""
        node_id = f"method:{method_name.lower().replace(' ', '_')}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(
                node_id,
                node_type=DSKGNodeType.METHOD.value,
                name=method_name,
                description=description,
                first_appeared_year=year,
            )
        return node_id

    def add_finding_node(self, finding_text: str, paper_id: str = "", year: int = 0) -> str:
        """Add a finding/result node."""
        self._node_counter["finding"] += 1
        node_id = f"finding:F-{self._node_counter['finding']:04d}"
        self.graph.add_node(
            node_id,
            node_type=DSKGNodeType.FINDING.value,
            text=finding_text[:500],
            source_paper_id=paper_id,
            year=year,
        )
        return node_id

    # ─── Edge Operations ────────────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: DSKGEdgeType,
        weight: float = 1.0,
        metadata: dict | None = None,
    ):
        """Add a typed epistemic edge between two nodes."""
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            logger.warning(f"Cannot add edge: missing node(s) {source_id} -> {target_id}")
            return

        edge_data = {
            "edge_type": edge_type.value,
            "weight": weight,
        }
        if metadata:
            edge_data.update(metadata)

        self.graph.add_edge(source_id, target_id, **edge_data)

        # contradicts is bidirectional
        if edge_type == DSKGEdgeType.CONTRADICTS:
            self.graph.add_edge(target_id, source_id, **edge_data)

    def add_supports_edge(self, claim_a_id: str, claim_b_id: str, weight: float = 1.0):
        """Claim A provides evidence for Claim B."""
        self.add_edge(f"claim:{claim_a_id}", f"claim:{claim_b_id}", DSKGEdgeType.SUPPORTS, weight)

    def add_contradicts_edge(self, claim_a_id: str, claim_b_id: str, weight: float = 1.0):
        """Claim A conflicts with Claim B (bidirectional)."""
        self.add_edge(f"claim:{claim_a_id}", f"claim:{claim_b_id}", DSKGEdgeType.CONTRADICTS, weight)

    def add_extends_edge(self, paper_a_id: str, paper_b_id: str):
        """Paper A builds upon Paper B."""
        self.add_edge(f"paper:{paper_a_id}", f"paper:{paper_b_id}", DSKGEdgeType.EXTENDS)

    def add_applies_to_edge(self, method_name: str, concept_name: str, paper_id: str = ""):
        """Method is applied to a domain/concept."""
        m_id = f"method:{method_name.lower().replace(' ', '_')}"
        c_id = f"concept:{concept_name.lower().replace(' ', '_')}"
        self.add_edge(m_id, c_id, DSKGEdgeType.APPLIES_TO, metadata={"paper_id": paper_id})

    # ─── Query Operations ───────────────────────────────────────────

    def get_nodes_by_type(self, node_type: DSKGNodeType) -> list[tuple[str, dict]]:
        """Get all nodes of a specific type."""
        return [
            (n, d)
            for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == node_type.value
        ]

    def get_edges_by_type(self, edge_type: DSKGEdgeType) -> list[tuple[str, str, dict]]:
        """Get all edges of a specific type."""
        return [
            (u, v, d)
            for u, v, d in self.graph.edges(data=True)
            if d.get("edge_type") == edge_type.value
        ]

    def get_claim_neighbors(self, claim_id: str) -> dict[str, list[str]]:
        """Get neighboring claims grouped by edge type."""
        node_id = f"claim:{claim_id}" if not claim_id.startswith("claim:") else claim_id
        result: dict[str, list[str]] = {}

        if not self.graph.has_node(node_id):
            return result

        for _, target, data in self.graph.edges(node_id, data=True):
            edge_type = data.get("edge_type", "unknown")
            result.setdefault(edge_type, []).append(target)

        for source, _, data in self.graph.in_edges(node_id, data=True):
            edge_type = data.get("edge_type", "unknown")
            result.setdefault(f"incoming_{edge_type}", []).append(source)

        return result

    def get_paper_claims(self, paper_id: str) -> list[tuple[str, dict]]:
        """Get all claim nodes introduced by a paper."""
        paper_node = f"paper:{paper_id}"
        claims = []
        if self.graph.has_node(paper_node):
            for _, target, data in self.graph.edges(paper_node, data=True):
                if data.get("edge_type") == DSKGEdgeType.INTRODUCES.value:
                    node_data = self.graph.nodes[target]
                    if node_data.get("node_type") == DSKGNodeType.CLAIM.value:
                        claims.append((target, node_data))
        return claims

    def get_contradiction_pairs(self) -> list[tuple[str, str, dict]]:
        """Get all contradiction edges."""
        return self.get_edges_by_type(DSKGEdgeType.CONTRADICTS)

    def get_subgraph(self, node_ids: list[str], depth: int = 1) -> nx.DiGraph:
        """Extract a subgraph centered on given nodes with specified depth."""
        all_nodes = set(node_ids)
        current_frontier = set(node_ids)

        for _ in range(depth):
            next_frontier = set()
            for node in current_frontier:
                if self.graph.has_node(node):
                    next_frontier.update(self.graph.successors(node))
                    next_frontier.update(self.graph.predecessors(node))
            all_nodes.update(next_frontier)
            current_frontier = next_frontier

        return self.graph.subgraph(all_nodes).copy()

    # ─── Statistics ─────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get summary statistics of the DSKG."""
        node_types = {}
        for _, data in self.graph.nodes(data=True):
            nt = data.get("node_type", "unknown")
            node_types[nt] = node_types.get(nt, 0) + 1

        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            et = data.get("edge_type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "is_connected": nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False,
            "num_components": nx.number_weakly_connected_components(self.graph) if self.graph.number_of_nodes() > 0 else 0,
        }

    # ─── Persistence ────────────────────────────────────────────────

    def save(self, path: str | None = None):
        """Save the DSKG to a JSON file."""
        save_path = path or self.persist_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        data = nx.node_link_data(self.graph)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"DSKG saved to {save_path} ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)")

    def load(self, path: str | None = None):
        """Load the DSKG from a JSON file."""
        load_path = path or self.persist_path
        if not Path(load_path).exists():
            logger.warning(f"No DSKG file found at {load_path}")
            return

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph = nx.node_link_graph(data, directed=True)
        logger.info(f"DSKG loaded from {load_path} ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)")

    def clear(self):
        """Clear the entire graph."""
        self.graph.clear()
        logger.info("DSKG cleared")
