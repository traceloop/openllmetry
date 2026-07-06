"""Unit tests for CAUSED_BY_GENERATION attribution links on tool call spans."""

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from opentelemetry.instrumentation.langchain.callback_handler import (
    TraceloopCallbackHandler,
)


@pytest.fixture
def _exporter():
    return InMemorySpanExporter()


@pytest.fixture
def _handler(_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_exporter))
    tracer = provider.get_tracer("test")

    reader = InMemoryMetricReader()
    meter = MeterProvider(metric_readers=[reader]).get_meter("test")
    duration_hist = meter.create_histogram("gen_ai.client.operation.duration")
    token_hist = meter.create_histogram("gen_ai.client.token.usage")
    return TraceloopCallbackHandler(tracer, duration_hist, token_hist)


def _llm_result_with_tool_call(tool_call_id: str, tool_name: str) -> LLMResult:
    msg = AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": tool_name, "args": {}}],
    )
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def test_caused_by_generation_link_structured_tool_call(_handler, _exporter):
    """Tool span must carry a CAUSED_BY_GENERATION link to the LLM span that triggered it."""
    root_id = uuid4()
    llm_id = uuid4()
    tool_id = uuid4()
    tc_id = "call_test_abc123"

    _handler.on_chain_start(
        {"id": ["FakeChain"], "name": "FakeChain"},
        {},
        run_id=root_id,
    )
    _handler.on_chat_model_start(
        {"id": ["FakeLLM"], "name": "FakeLLM"},
        [[]],
        run_id=llm_id,
        parent_run_id=root_id,
    )
    _handler.on_llm_end(
        _llm_result_with_tool_call(tc_id, "search"),
        run_id=llm_id,
        parent_run_id=root_id,
    )
    _handler.on_tool_start(
        {"id": ["SearchTool"], "name": "search"},
        "test query",
        run_id=tool_id,
        parent_run_id=root_id,
        tool_call_id=tc_id,
    )
    _handler.on_tool_end("result", run_id=tool_id, parent_run_id=root_id)
    _handler.on_chain_end({}, run_id=root_id)

    finished = _exporter.get_finished_spans()
    tool_spans = [s for s in finished if "execute_tool" in s.name]
    assert len(tool_spans) == 1, f"Unexpected spans: {[s.name for s in finished]}"

    tool_span = tool_spans[0]
    assert len(tool_span.links) == 1, "Tool span must have exactly one attribution link"

    link = tool_span.links[0]
    assert link.attributes.get("gen_ai.attribution.link_type") == "CAUSED_BY_GENERATION"

    # Link must point to the LLM span's context
    llm_spans = [s for s in finished if ".chat" in s.name]
    assert len(llm_spans) == 1
    llm_span = llm_spans[0]
    assert link.context.trace_id == llm_span.context.trace_id
    assert link.context.span_id == llm_span.context.span_id


def test_no_attribution_link_without_tool_call_id(_handler, _exporter):
    """Tool span without a structured tool_call_id must carry no attribution link."""
    root_id = uuid4()
    tool_id = uuid4()

    _handler.on_chain_start(
        {"id": ["FakeChain"], "name": "FakeChain"},
        {},
        run_id=root_id,
    )
    _handler.on_tool_start(
        {"id": ["SomeTool"], "name": "mytool"},
        "input",
        run_id=tool_id,
        parent_run_id=root_id,
    )
    _handler.on_tool_end("output", run_id=tool_id, parent_run_id=root_id)
    _handler.on_chain_end({}, run_id=root_id)

    finished = _exporter.get_finished_spans()
    tool_spans = [s for s in finished if "execute_tool" in s.name]
    assert len(tool_spans) == 1
    assert len(tool_spans[0].links) == 0, "Tool span with no LLM predecessor must have no links"
