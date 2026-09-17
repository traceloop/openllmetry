"""Tests for the tool-execution span emitted by wrap_tool_run (issue #4452).

Uses a real TracerProvider + InMemorySpanExporter so we assert the attributes
that actually land on the emitted span, following RFC #3460 (gen_ai.tool.name,
gen_ai.tool.type, gen_ai.operation.name = execute_tool).
"""

import json

import pytest
from crewai.tools import BaseTool
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiOperationNameValues,
)
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import StatusCode

from opentelemetry.instrumentation.crewai.instrumentation import wrap_tool_run


class EchoTool(BaseTool):
    name: str = "echo"
    description: str = "Echoes the input back."

    def _run(self, text: str = "") -> str:
        return f"echoed: {text}"


class BoomTool(BaseTool):
    name: str = "boom"
    description: str = "Always raises."

    def _run(self, **kwargs) -> str:
        raise ValueError("kaboom")


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test")


def _first_span(exporter):
    spans = exporter.get_finished_spans()
    assert spans, "no finished spans"
    return spans[0]


def test_tool_run_emits_span_with_semconv_attributes(tracer, exporter):
    tool = EchoTool()

    result = wrap_tool_run(tracer, None, None)(
        tool.run, tool, [], {"text": "hi"}
    )

    assert result == "echoed: hi"
    span = _first_span(exporter)
    attrs = dict(span.attributes or {})

    assert span.name == "echo.tool"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.OK
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == GenAiOperationNameValues.EXECUTE_TOOL.value
    assert attrs[GenAIAttributes.GEN_AI_TOOL_NAME] == "echo"
    assert attrs[GenAIAttributes.GEN_AI_TOOL_TYPE] == "function"
    assert attrs[GenAIAttributes.GEN_AI_TOOL_DESCRIPTION] == "Echoes the input back."


def test_tool_run_captures_arguments_and_result(tracer, exporter):
    tool = EchoTool()

    wrap_tool_run(tracer, None, None)(tool.run, tool, [], {"text": "hi"})

    attrs = dict(_first_span(exporter).attributes or {})
    assert json.loads(attrs["gen_ai.tool.call.arguments"]) == {"text": "hi"}
    assert attrs["gen_ai.tool.call.result"] == "echoed: hi"


def test_tool_run_records_error_status_and_reraises(tracer, exporter):
    tool = BoomTool()

    with pytest.raises(ValueError, match="kaboom"):
        wrap_tool_run(tracer, None, None)(tool.run, tool, [], {})

    span = _first_span(exporter)
    assert span.name == "boom.tool"
    assert span.status.status_code == StatusCode.ERROR
