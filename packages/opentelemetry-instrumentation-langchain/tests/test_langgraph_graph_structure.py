"""Tests for LangGraph graph-structure extraction.

Covers traceloop/openllmetry#4447: LangGraph permits str-subclass node ids
(e.g. ``StrEnum`` members). ``extract_graph_structure`` appended them raw, and
OpenTelemetry validates sequence attributes with an exact-type check, so the
whole ``gen_ai.workflow.nodes`` attribute was rejected (and a warning logged)
on every graph invocation.
"""

from enum import StrEnum
from typing import NamedTuple

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes

from opentelemetry.instrumentation.langchain.langgraph_utils import (
    extract_graph_structure,
)


class NodeName(StrEnum):
    PARSE = "parse"
    CLASSIFY = "classify"


class Edge(NamedTuple):
    source: str
    target: str


class FakeGraph:
    """Minimal stand-in for LangGraph's drawable graph (nodes dict + edges)."""

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges


class FakeCompiledGraph:
    def __init__(self, graph):
        self._graph = graph

    def get_graph(self):
        return self._graph


def _record_attributes(nodes, edges):
    """Run the values through a real OTel span to trigger attribute validation."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("graph") as span:
        span.set_attribute(SpanAttributes.GEN_AI_WORKFLOW_NODES, nodes)
        span.set_attribute(SpanAttributes.GEN_AI_WORKFLOW_EDGES, edges)
    provider.shutdown()
    attributes = exporter.get_finished_spans()[0].attributes
    return attributes


def test_str_subclass_node_ids_are_coerced_to_plain_str():
    graph = FakeCompiledGraph(
        FakeGraph(
            nodes={
                "__start__": None,
                NodeName.PARSE: None,
                NodeName.CLASSIFY: None,
                "__end__": None,
            },
            edges=[
                Edge("__start__", NodeName.PARSE),
                Edge(NodeName.PARSE, NodeName.CLASSIFY),
                Edge(NodeName.CLASSIFY, "__end__"),
            ],
        )
    )

    nodes, edges = extract_graph_structure(graph)

    # Special nodes are excluded and every value is a plain str (exact type),
    # not a str subclass such as StrEnum.
    assert nodes == ["parse", "classify"]
    assert all(type(node) is str for node in nodes)
    assert edges == ["parse -> classify"]
    assert all(type(edge) is str for edge in edges)


def test_attributes_accepted_by_otel_with_str_subclass_ids():
    graph = FakeCompiledGraph(
        FakeGraph(
            nodes={NodeName.PARSE: None, NodeName.CLASSIFY: None},
            edges=[Edge(NodeName.PARSE, NodeName.CLASSIFY)],
        )
    )
    nodes, edges = extract_graph_structure(graph)

    attributes = _record_attributes(nodes, edges)

    # Before the fix the nodes attribute was rejected entirely (None) and
    # OpenTelemetry logged an "Invalid type" warning.
    assert attributes[SpanAttributes.GEN_AI_WORKFLOW_NODES] == ("parse", "classify")
    assert attributes[SpanAttributes.GEN_AI_WORKFLOW_EDGES] == ("parse -> classify",)


def test_plain_string_node_ids_unchanged():
    graph = FakeCompiledGraph(
        FakeGraph(
            nodes={
                "__start__": None,
                "calculate": None,
                "report": None,
                "__end__": None,
            },
            edges=[
                Edge("__start__", "calculate"),
                Edge("calculate", "__end__"),
                Edge("calculate", "report"),
            ],
        )
    )

    nodes, edges = extract_graph_structure(graph)

    assert nodes == ["calculate", "report"]
    # Edges touching __start__/__end__ are skipped; the user-facing edge remains.
    assert edges == ["calculate -> report"]
    assert all(type(edge) is str for edge in edges)
