"""Utilities for extracting LangGraph graph structure."""

import json
from typing import Any


def extract_graph_json(graph_instance: Any) -> str:
    """
    Extract the graph structure (nodes and edges) from a LangGraph instance.

    This extracts the workflow topology to populate the gen_ai.workflow.structure
    attribute as specified in the OpenTelemetry GenAI semantic conventions.

    Args:
        graph_instance: A LangGraph Pregel or CompiledGraph instance

    Returns:
        JSON string representing the graph structure with format:
        {
            "nodes": ["node1", "node2", ...],
            "edges": [
                [["source1"], ["target1"]],
                [["source2"], ["target2", "target3"]],
                ...
            ]
        }

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

        # Extract edges
        edges = []
        if hasattr(graph, "edges"):
            # Group edges by source for cleaner representation
            edge_map: dict[str, list[str]] = {}
            for edge in graph.edges:
                # Handle different edge formats
                if hasattr(edge, "source") and hasattr(edge, "target"):
                    source = edge.source
                    target = edge.target
                elif isinstance(edge, tuple) and len(edge) >= 2:
                    source, target = edge[0], edge[1]
                else:
                    continue

                if source not in edge_map:
                    edge_map[source] = []
                if target not in edge_map[source]:
                    edge_map[source].append(target)

            # Convert to list format [[sources], [targets]]
            for source, targets in edge_map.items():
                edges.append([[source], targets])

        structure = {
            "nodes": nodes,
            "edges": edges,
        }

        return json.dumps(structure)

    except Exception as e:
        # Re-raise to let caller handle
        raise Exception(f"Failed to extract graph structure: {e}") from e
