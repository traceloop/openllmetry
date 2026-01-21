import inspect
import json
from langgraph.graph.state import CompiledStateGraph


from typing import List, Dict, Union, Any


def is_langgraph_task(name: str) -> bool:
    return name == "LangGraph"


def get_compiled_graph():
    """ Get the compiled graph from the call stack """
    graph = None
    invocation_methods = ["Pregel.invoke", "Pregel.ainvoke", "Pregel.stream", "Pregel.astream"]
    frames = inspect.stack()
    for frame_info in frames[1:]:
        if frame_info.frame.f_code.co_qualname in invocation_methods:
            local_vars = frame_info.frame.f_locals
            graph = local_vars.get("self", None)
            graph = graph if isinstance(graph, CompiledStateGraph) else None
            break
    return graph


def _normalize_endpoint_names(
    names: Union[str, List[str], tuple[str, ...]]
) -> List[str]:
    """Normalize edge endpoints to a list of node names."""
    if isinstance(names, str):
        return [names]
    if isinstance(names, (list, tuple)):
        return list(names)
    raise TypeError(f"Unsupported endpoint type: {type(names)!r}")


def build_node_graph(compiled_state_graph: CompiledStateGraph) -> Dict[str, Any]:
    """
    Build a simple node/edge representation from CompiledStateGraph.

    Returns a dict:
    {
        "nodes": [node_name, ...],                  # excluding "__start__", "__end__"
        "edges": [
            [[source1, ...], [dest1, dest2, ...]],  # each edge has list of sources and list of destinations
            ...
        ]
    }
    """
    builder = compiled_state_graph.builder

    # Track *all* node names (including __start__/__end__) for edges,
    # but only expose non-special nodes in the "nodes" list.
    all_nodes_ordered = list(compiled_state_graph.nodes.keys())
    nodes: List[str] = [
        name for name in all_nodes_ordered if name not in ("__start__", "__end__")
    ]

    edges: List[List[List[str]]] = []

    # Regular edges
    for src, dst in builder.edges:
        src_names = _normalize_endpoint_names(src)
        dst_names = _normalize_endpoint_names(dst)
        edges.append([src_names, dst_names])

    # Branches
    branches: Dict[str, Dict[str]] = builder.branches
    for source, branch_map in branches.items():
        for branch in branch_map.values():
            # branch.ends is expected to be a mapping; we use its values as destinations
            dest_names = list(branch.ends.values())
            # Source is a single node here
            edges.append([[source], dest_names])

    # Waiting edges
    for src, dst in builder.waiting_edges:
        src_names = _normalize_endpoint_names(src)
        dst_names = _normalize_endpoint_names(dst)
        edges.append([src_names, dst_names])

    return {
        "nodes": nodes,
        "edges": edges,
    }


def get_graph_structure() -> str:
    """
    Get graph structure as a JSON string.

    Returns:
        JSON string with structure:
        {
            "nodes": [...],
            "edges": [[[...], [...]], ...]
        }
    """
    graph_structure: Dict[str, Any] = {}
    graph = get_compiled_graph()
    if graph:
        graph_structure = build_node_graph(graph)
    return json.dumps(graph_structure)
