"""
Graph algorithms for the DSKG — §4.5.3 and §5.2.3.
Implements Louvain community detection, centrality analysis, lineage tracing,
cross-cluster analysis, and structural gap detection.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import networkx as nx
from loguru import logger

from src.models.data_models import DSKGEdgeType, DSKGNodeType


def detect_communities(graph: nx.DiGraph) -> dict[str, int]:
    """
    Run Louvain community detection for thematic clustering (§4.5.3).
    Returns mapping of node_id -> community_id.
    """
    try:
        import community as community_louvain
    except ImportError:
        logger.warning("python-louvain not installed. Using connected components as fallback.")
        communities = {}
        for i, component in enumerate(nx.weakly_connected_components(graph)):
            for node in component:
                communities[node] = i
        return communities

    # Louvain works on undirected graphs
    undirected = graph.to_undirected()

    # Remove isolated nodes for cleaner communities
    non_isolated = [n for n in undirected.nodes() if undirected.degree(n) > 0]
    if not non_isolated:
        return {n: 0 for n in graph.nodes()}

    subgraph = undirected.subgraph(non_isolated)
    partition = community_louvain.best_partition(subgraph)

    # Assign isolated nodes to community -1
    for n in graph.nodes():
        if n not in partition:
            partition[n] = -1

    num_communities = len(set(partition.values()) - {-1})
    logger.info(f"Detected {num_communities} communities via Louvain")
    return partition


def compute_centrality(graph: nx.DiGraph) -> dict[str, dict[str, float]]:
    """
    Compute multiple centrality measures for DSKG nodes.
    Returns dict of node_id -> {measure_name: score}.
    """
    results = {}

    if graph.number_of_nodes() == 0:
        return results

    # PageRank — importance based on incoming edges
    try:
        pagerank = nx.pagerank(graph, alpha=0.85)
    except Exception:
        pagerank = {n: 0.0 for n in graph.nodes()}

    # Betweenness centrality — bridge nodes
    try:
        betweenness = nx.betweenness_centrality(graph, k=min(100, graph.number_of_nodes()))
    except Exception:
        betweenness = {n: 0.0 for n in graph.nodes()}

    # Degree centrality
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())

    for node in graph.nodes():
        results[node] = {
            "pagerank": pagerank.get(node, 0.0),
            "betweenness": betweenness.get(node, 0.0),
            "in_degree": in_degree.get(node, 0),
            "out_degree": out_degree.get(node, 0),
        }

    return results


def trace_lineage(graph: nx.DiGraph, node_id: str, direction: str = "both") -> list[list[str]]:
    """
    Trace the lineage of a node by following 'extends' and 'introduces' edges (§4.5.3).
    Returns lists of lineage paths.
    
    Args:
        graph: The DSKG graph
        node_id: Starting node
        direction: 'ancestors', 'descendants', or 'both'
    """
    paths = []

    if not graph.has_node(node_id):
        return paths

    if direction in ("ancestors", "both"):
        # Trace backwards via extends edges
        visited = set()
        stack = [(node_id, [node_id])]
        while stack:
            current, path = stack.pop()
            for pred in graph.predecessors(current):
                edge_data = graph[pred][current]
                if edge_data.get("edge_type") in (DSKGEdgeType.EXTENDS.value, DSKGEdgeType.INTRODUCES.value):
                    if pred not in visited:
                        visited.add(pred)
                        new_path = [pred] + path
                        paths.append(new_path)
                        stack.append((pred, new_path))

    if direction in ("descendants", "both"):
        # Trace forward
        visited = set()
        stack = [(node_id, [node_id])]
        while stack:
            current, path = stack.pop()
            for succ in graph.successors(current):
                edge_data = graph[current][succ]
                if edge_data.get("edge_type") in (DSKGEdgeType.EXTENDS.value, DSKGEdgeType.INTRODUCES.value):
                    if succ not in visited:
                        visited.add(succ)
                        new_path = path + [succ]
                        paths.append(new_path)
                        stack.append((succ, new_path))

    return paths


def compute_cross_cluster_density(
    graph: nx.DiGraph,
    communities: dict[str, int],
) -> dict[tuple[int, int], float]:
    """
    Compute edge density between community pairs (§5.2.3).
    Sparse cross-cluster connections indicate intersection gaps.
    """
    cluster_pairs: dict[tuple[int, int], int] = defaultdict(int)
    cluster_sizes: dict[int, int] = defaultdict(int)

    for node, comm_id in communities.items():
        cluster_sizes[comm_id] += 1

    for u, v, _ in graph.edges(data=True):
        comm_u = communities.get(u, -1)
        comm_v = communities.get(v, -1)
        if comm_u != comm_v and comm_u != -1 and comm_v != -1:
            pair = (min(comm_u, comm_v), max(comm_u, comm_v))
            cluster_pairs[pair] += 1

    # Normalize by potential edges between clusters
    density = {}
    for (c1, c2), edge_count in cluster_pairs.items():
        max_edges = cluster_sizes[c1] * cluster_sizes[c2]
        density[(c1, c2)] = edge_count / max_edges if max_edges > 0 else 0.0

    return density


def find_structural_gaps(
    graph: nx.DiGraph,
    communities: dict[str, int],
    centrality: dict[str, dict[str, float]],
) -> list[dict]:
    """
    Identify structural indicators of research gaps in the DSKG (§5.2.3).
    
    Returns list of gap indicators with type and evidence.
    """
    gaps = []

    # ── 1. Sparse cross-cluster connections → Intersection Gaps ──
    cross_density = compute_cross_cluster_density(graph, communities)
    for (c1, c2), density in cross_density.items():
        if density < 0.05:  # Very sparse connection
            # Get representative nodes from each cluster
            c1_nodes = [n for n, c in communities.items() if c == c1]
            c2_nodes = [n for n, c in communities.items() if c == c2]

            # Get topic labels from concept nodes
            c1_concepts = [
                graph.nodes[n].get("name", n)
                for n in c1_nodes
                if graph.nodes[n].get("node_type") == DSKGNodeType.CONCEPT.value
            ][:3]
            c2_concepts = [
                graph.nodes[n].get("name", n)
                for n in c2_nodes
                if graph.nodes[n].get("node_type") == DSKGNodeType.CONCEPT.value
            ][:3]

            gaps.append({
                "type": "intersection_gap",
                "clusters": (c1, c2),
                "density": density,
                "cluster_a_topics": c1_concepts,
                "cluster_b_topics": c2_concepts,
                "evidence": f"Cross-cluster edge density between clusters {c1} and {c2} is {density:.4f}",
            })

    # ── 2. High centrality + low extensions → Underexplored Gaps ──
    for node, scores in centrality.items():
        node_data = graph.nodes.get(node, {})
        if node_data.get("node_type") not in (DSKGNodeType.CONCEPT.value, DSKGNodeType.METHOD.value):
            continue

        # Count extends edges
        extends_count = sum(
            1
            for _, _, d in graph.edges(node, data=True)
            if d.get("edge_type") == DSKGEdgeType.EXTENDS.value
        )
        extends_in = sum(
            1
            for _, _, d in graph.in_edges(node, data=True)
            if d.get("edge_type") == DSKGEdgeType.EXTENDS.value
        )

        if scores["pagerank"] > 0.01 and (extends_count + extends_in) < 2:
            gaps.append({
                "type": "underexplored_gap",
                "node": node,
                "name": node_data.get("name", node),
                "pagerank": scores["pagerank"],
                "extends_count": extends_count + extends_in,
                "evidence": f"Node '{node_data.get('name', node)}' has high centrality "
                           f"(PageRank={scores['pagerank']:.4f}) but few extensions ({extends_count + extends_in})",
            })

    # ── 3. Dense contradictions, no resolution → Contradictory State Gaps ──
    contradiction_clusters: dict[int, int] = defaultdict(int)
    for u, v, data in graph.edges(data=True):
        if data.get("edge_type") == DSKGEdgeType.CONTRADICTS.value:
            comm = communities.get(u, -1)
            if comm != -1:
                contradiction_clusters[comm] += 1

    for comm_id, count in contradiction_clusters.items():
        if count >= 3:  # Dense contradictions
            comm_nodes = [n for n, c in communities.items() if c == comm_id]
            concepts = [
                graph.nodes[n].get("name", n)
                for n in comm_nodes
                if graph.nodes[n].get("node_type") == DSKGNodeType.CONCEPT.value
            ][:3]

            gaps.append({
                "type": "contradictory_state_gap",
                "cluster": comm_id,
                "contradiction_count": count,
                "topics": concepts,
                "evidence": f"Cluster {comm_id} has {count} contradiction edges with no resolution",
            })

    # ── 4. Methods applied to few domains → Methodological Gaps ──
    for node, data in graph.nodes(data=True):
        if data.get("node_type") != DSKGNodeType.METHOD.value:
            continue

        applied_domains = set()
        for _, target, edge_data in graph.edges(node, data=True):
            if edge_data.get("edge_type") == DSKGEdgeType.APPLIES_TO.value:
                applied_domains.add(target)

        if 0 < len(applied_domains) < 2:
            gaps.append({
                "type": "methodological_gap",
                "method_node": node,
                "method_name": data.get("name", node),
                "applied_to": list(applied_domains),
                "evidence": f"Method '{data.get('name', node)}' applied to only {len(applied_domains)} domain(s)",
            })

    logger.info(f"Found {len(gaps)} structural gap indicators in DSKG")
    return gaps


def get_cluster_summary(
    graph: nx.DiGraph,
    communities: dict[str, int],
) -> dict[int, dict]:
    """Get a summary of each community cluster including nodes, edges, and key concepts."""
    summaries: dict[int, dict] = {}

    for comm_id in set(communities.values()):
        if comm_id == -1:
            continue

        members = [n for n, c in communities.items() if c == comm_id]
        subgraph = graph.subgraph(members)

        # Count node types
        node_types = defaultdict(int)
        concepts = []
        methods = []
        claims = []

        for node in members:
            data = graph.nodes.get(node, {})
            nt = data.get("node_type", "unknown")
            node_types[nt] += 1

            if nt == DSKGNodeType.CONCEPT.value:
                concepts.append(data.get("name", node))
            elif nt == DSKGNodeType.METHOD.value:
                methods.append(data.get("name", node))
            elif nt == DSKGNodeType.CLAIM.value:
                claims.append(data.get("claim_text", "")[:100])

        summaries[comm_id] = {
            "size": len(members),
            "node_types": dict(node_types),
            "concepts": concepts[:10],
            "methods": methods[:10],
            "sample_claims": claims[:5],
            "internal_edges": subgraph.number_of_edges(),
        }

    return summaries
