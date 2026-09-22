from types import SimpleNamespace

from opentelemetry.instrumentation.langchain.langgraph_utils import extract_graph_structure
from opentelemetry.sdk.trace import TracerProvider


class NodeId(str):
    """Represent a graph node ID with a custom string type."""
    pass


def test_string_subclasses_are_exported_with_standard_strings():
    """Verify mixed string types survive extraction and span attribute validation."""
    graph = SimpleNamespace(
        nodes=[NodeId("__start__"), "plain", NodeId("custom"), NodeId("__end__")],
        edges=[("plain", NodeId("custom")), (NodeId("__start__"), "plain")],
    )
    nodes, edges = extract_graph_structure(graph)
    assert nodes == ["plain", "custom"]
    assert edges == ["plain -> custom"]
    provider = TracerProvider()
    with provider.get_tracer(__name__).start_as_current_span("graph") as span:
        span.set_attribute("gen_ai.workflow.nodes", nodes)
        assert span.attributes["gen_ai.workflow.nodes"] == ("plain", "custom")
    provider.shutdown()
