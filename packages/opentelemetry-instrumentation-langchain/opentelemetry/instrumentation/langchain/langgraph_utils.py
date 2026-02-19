"""Utilities for extracting LangGraph graph structure."""

from typing import Any


def extract_graph_structure(graph_instance: Any) -> tuple[list[str], list[str]]:
    """
    Extract graph nodes and edges as separate lists.

    This extracts the workflow topology to populate the gen_ai.workflow.nodes
    and gen_ai.workflow.edges attributes as specified in the OpenTelemetry
    GenAI semantic conventions.

    Args:
        graph_instance: A LangGraph Pregel or CompiledGraph instance

    Returns:
        Tuple of (nodes, edges) where:
        - nodes: List of node names (excluding __start__/__end__)
        - edges: List of "source -> target" strings

    Raises:
        Exception: If graph structure cannot be extracted
    """
    try:
        # Try to get the graph structure via get_graph() method
        if hasattr(graph_instance, "get_graph"):
            graph = graph_instance.get_graph()
        else:
            graph = graph_instance

        # Extract nodes (excluding __start__ and __end__ special nodes)
        nodes = []
        if hasattr(graph, "nodes"):
            for node_id in graph.nodes:
                if node_id not in ("__start__", "__end__"):
                    nodes.append(node_id)

        # Extract edges as "source -> target" strings
        edges = []
        if hasattr(graph, "edges"):
            for edge in graph.edges:
                # Handle different edge formats
                if hasattr(edge, "source") and hasattr(edge, "target"):
                    source, target = edge.source, edge.target
                elif isinstance(edge, tuple) and len(edge) >= 2:
                    source, target = edge[0], edge[1]
                else:
                    continue

                # Skip special nodes
                if source not in ("__start__", "__end__") and target not in ("__start__", "__end__"):
                    edges.append(f"{source} -> {target}")

        return nodes, edges

    except Exception as e:
        # Re-raise to let caller handle
        raise Exception(f"Failed to extract graph structure: {e}") from e
