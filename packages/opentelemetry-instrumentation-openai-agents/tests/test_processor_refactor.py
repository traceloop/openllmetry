"""
TDD tests for on_span_start / on_span_end refactoring.

Tests target the extracted helper methods:
  on_span_start handlers:
    _start_agent_span, _start_handoff_span, _start_function_span,
    _start_generation_span, _start_realtime_span
  on_span_end helpers:
    _extract_tool_definitions  (pure function)
    _end_generation_span       (method)
    _set_realtime_io_attributes (method)

These tests are written BEFORE the implementation (TDD).
"""

import json
import pytest
from unittest.mock import MagicMock
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import SpanKind
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracer_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test-refactor"), exporter


@pytest.fixture
def processor(tracer_and_exporter):
    from opentelemetry.instrumentation.openai_agents._hooks import (
        OpenTelemetryTracingProcessor,
    )
    tracer, _ = tracer_and_exporter
    return OpenTelemetryTracingProcessor(tracer)


# ---------------------------------------------------------------------------
# Helpers: mock SDK span_data objects
# ---------------------------------------------------------------------------

class MockAgentSpan:
    """Minimal mock of an Agents SDK span object."""
    def __init__(self, span_data, trace_id="test-trace", error=None):
        self.span_data = span_data
        self.trace_id = trace_id
        self.error = error


# ---------------------------------------------------------------------------
# Tests: _start_agent_span
# ---------------------------------------------------------------------------

class TestStartAgentSpan:
    """Unit tests for the extracted _start_agent_span handler."""

    def test_returns_span_with_agent_attributes(self, tracer_and_exporter, processor):
        """Must return a span named '{name}.agent' with correct attributes."""
        from agents import AgentSpanData

        tracer, exporter = tracer_and_exporter
        agent_data = AgentSpanData(name="MyAgent", handoffs=[], tools=[], output_type="")

        otel_span = processor._start_agent_span(agent_data, parent_context=None, trace_id="t1")

        assert otel_span is not None
        assert otel_span.name == "MyAgent.agent"
        attrs = dict(otel_span.attributes)
        assert attrs[SpanAttributes.TRACELOOP_SPAN_KIND] == TraceloopSpanKindValues.AGENT.value
        assert attrs[GenAIAttributes.GEN_AI_AGENT_NAME] == "MyAgent"
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"

        otel_span.end()

    def test_unknown_agent_name_defaults(self, tracer_and_exporter, processor):
        """Agent with no name → 'unknown_agent'."""
        from agents import AgentSpanData

        agent_data = AgentSpanData(name=None, handoffs=[], tools=[], output_type="")

        otel_span = processor._start_agent_span(agent_data, parent_context=None, trace_id="t2")

        assert otel_span.name == "unknown_agent.agent"
        assert otel_span.attributes[GenAIAttributes.GEN_AI_AGENT_NAME] == "unknown_agent"

        otel_span.end()

    def test_handoff_parent_attribute_set(self, tracer_and_exporter, processor):
        """When a reverse handoff exists, handoff_parent must be set."""
        from agents import AgentSpanData

        # Pre-seed the reverse handoff dict
        processor._reverse_handoffs_dict["TargetAgent:t3"] = "SourceAgent"

        agent_data = AgentSpanData(name="TargetAgent", handoffs=[], tools=[], output_type="")
        otel_span = processor._start_agent_span(agent_data, parent_context=None, trace_id="t3")

        attrs = dict(otel_span.attributes)
        assert attrs.get("gen_ai.agent.handoff_parent") == "SourceAgent"
        # Consumed from the dict
        assert "TargetAgent:t3" not in processor._reverse_handoffs_dict

        otel_span.end()

    def test_handoffs_list_serialized(self, tracer_and_exporter, processor):
        """Handoff targets should be serialized as JSON attributes."""
        from agents import AgentSpanData

        mock_handoff_agent = MagicMock()
        mock_handoff_agent.name = "AgentB"
        mock_handoff_agent.instructions = "Help the user"

        agent_data = AgentSpanData(name="AgentA", handoffs=[mock_handoff_agent], tools=[], output_type="")
        otel_span = processor._start_agent_span(agent_data, parent_context=None, trace_id="t4")

        attrs = dict(otel_span.attributes)
        handoff_json = json.loads(attrs["openai.agent.handoff0"])
        assert handoff_json["name"] == "AgentB"
        assert handoff_json["instructions"] == "Help the user"

        otel_span.end()

    def test_span_kind_is_internal(self, tracer_and_exporter, processor):
        """Agent spans must be INTERNAL kind (in-process orchestration, not a remote call)."""
        from agents import AgentSpanData

        agent_data = AgentSpanData(name="Agent", handoffs=[], tools=[], output_type="")
        otel_span = processor._start_agent_span(agent_data, parent_context=None, trace_id="t5")

        assert otel_span.kind == SpanKind.INTERNAL

        otel_span.end()


# ---------------------------------------------------------------------------
# Tests: _start_handoff_span
# ---------------------------------------------------------------------------

class TestStartHandoffSpan:
    """Unit tests for the extracted _start_handoff_span handler."""

    def test_returns_span_with_handoff_attributes(self, tracer_and_exporter, processor):
        """Must create a span named '{from} → {to}.handoff'."""
        from agents import HandoffSpanData

        handoff_data = HandoffSpanData(from_agent="AgentA", to_agent="AgentB")

        otel_span = processor._start_handoff_span(
            handoff_data, parent_context=None, trace_id="t1",
        )

        assert otel_span is not None
        assert otel_span.name == "AgentA → AgentB.handoff"
        attrs = dict(otel_span.attributes)
        assert attrs[SpanAttributes.TRACELOOP_SPAN_KIND] == "handoff"
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"

        otel_span.end()

    def test_from_and_to_agent_attributes(self, tracer_and_exporter, processor):
        """from_agent and to_agent must be set as attributes."""
        from agents import HandoffSpanData
        from opentelemetry.instrumentation.openai_agents.utils import (
            GEN_AI_HANDOFF_FROM_AGENT,
            GEN_AI_HANDOFF_TO_AGENT,
        )

        handoff_data = HandoffSpanData(from_agent="AgentA", to_agent="AgentB")

        otel_span = processor._start_handoff_span(
            handoff_data, parent_context=None, trace_id="t2",
        )

        attrs = dict(otel_span.attributes)
        assert attrs[GEN_AI_HANDOFF_FROM_AGENT] == "AgentA"
        assert attrs[GEN_AI_HANDOFF_TO_AGENT] == "AgentB"

        otel_span.end()

    def test_registers_reverse_handoff(self, tracer_and_exporter, processor):
        """Must register reverse handoff for the target agent."""
        from agents import HandoffSpanData

        handoff_data = HandoffSpanData(from_agent="AgentA", to_agent="AgentB")

        processor._start_handoff_span(
            handoff_data, parent_context=None, trace_id="trace-123",
        )

        assert processor._reverse_handoffs_dict.get("AgentB:trace-123") == "AgentA"

    def test_unknown_agents_fallback(self, tracer_and_exporter, processor):
        """None agent names → 'unknown' in span name."""
        from agents import HandoffSpanData

        handoff_data = HandoffSpanData(from_agent=None, to_agent=None)

        otel_span = processor._start_handoff_span(
            handoff_data, parent_context=None, trace_id="t3",
        )

        assert "unknown" in otel_span.name

        otel_span.end()

    def test_span_kind_is_internal(self, tracer_and_exporter, processor):
        """Handoff spans must be INTERNAL kind."""
        from agents import HandoffSpanData

        handoff_data = HandoffSpanData(from_agent="A", to_agent="B")
        otel_span = processor._start_handoff_span(
            handoff_data, parent_context=None, trace_id="t4",
        )

        assert otel_span.kind == SpanKind.INTERNAL

        otel_span.end()


# ---------------------------------------------------------------------------
# Tests: _start_function_span
# ---------------------------------------------------------------------------

class TestStartFunctionSpan:
    """Unit tests for the extracted _start_function_span handler."""

    def test_returns_span_with_tool_attributes(self, tracer_and_exporter, processor):
        """Must return a span named '{tool}.tool' with tool attributes."""
        from agents import FunctionSpanData

        func_data = FunctionSpanData(name="get_weather", input="", output="")

        otel_span = processor._start_function_span(func_data, parent_context=None)

        assert otel_span is not None
        assert otel_span.name == "get_weather.tool"
        attrs = dict(otel_span.attributes)
        assert attrs[GenAIAttributes.GEN_AI_TOOL_NAME] == "get_weather"
        assert attrs[GenAIAttributes.GEN_AI_TOOL_TYPE] == "function"
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"
        assert attrs[SpanAttributes.TRACELOOP_SPAN_KIND] == TraceloopSpanKindValues.TOOL.value

        otel_span.end()

    def test_unknown_tool_name_defaults(self, tracer_and_exporter, processor):
        """Tool with no name → 'unknown_tool'."""
        from agents import FunctionSpanData

        func_data = FunctionSpanData(name=None, input="", output="")

        otel_span = processor._start_function_span(func_data, parent_context=None)

        assert otel_span.name == "unknown_tool.tool"

        otel_span.end()

    def test_description_attribute_set(self, tracer_and_exporter, processor):
        """Non-generic descriptions must appear as GEN_AI_TOOL_DESCRIPTION."""
        from agents import FunctionSpanData

        func_data = FunctionSpanData(name="search", input="", output="")
        func_data.description = "Search the web for information"

        otel_span = processor._start_function_span(func_data, parent_context=None)

        attrs = dict(otel_span.attributes)
        assert attrs[GenAIAttributes.GEN_AI_TOOL_DESCRIPTION] == "Search the web for information"

        otel_span.end()

    def test_generic_description_filtered_out(self, tracer_and_exporter, processor):
        """Descriptions starting with 'Represents a Function Span' must be ignored."""
        from agents import FunctionSpanData

        func_data = FunctionSpanData(name="search", input="", output="")
        func_data.description = "Represents a Function Span for search"

        otel_span = processor._start_function_span(func_data, parent_context=None)

        attrs = dict(otel_span.attributes)
        assert GenAIAttributes.GEN_AI_TOOL_DESCRIPTION not in attrs

        otel_span.end()

    def test_span_kind_is_internal(self, tracer_and_exporter, processor):
        """Function/tool spans must be INTERNAL kind."""
        from agents import FunctionSpanData

        func_data = FunctionSpanData(name="tool", input="", output="")
        otel_span = processor._start_function_span(func_data, parent_context=None)

        assert otel_span.kind == SpanKind.INTERNAL

        otel_span.end()


# ---------------------------------------------------------------------------
# Tests: _end_function_span
# ---------------------------------------------------------------------------

class TestEndFunctionSpan:
    """Unit tests for _end_function_span — sets tool call arguments/result."""

    def test_sets_tool_call_arguments_and_result(self, tracer_and_exporter, processor):
        """Must set gen_ai.tool.call.arguments and gen_ai.tool.call.result."""
        from agents import FunctionSpanData

        func_data = FunctionSpanData(
            name="get_weather", input='{"city": "NYC"}', output='{"temp": 72}'
        )
        otel_span = processor._start_function_span(func_data, parent_context=None)
        processor._end_function_span(otel_span, func_data, trace_content=True)
        otel_span.end()

        attrs = dict(otel_span.attributes)
        assert attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS] == '{"city": "NYC"}'
        assert attrs[GenAIAttributes.GEN_AI_TOOL_CALL_RESULT] == '{"temp": 72}'

    def test_content_gated_when_false(self, tracer_and_exporter, processor):
        """Must NOT set arguments/result when trace_content is False."""
        from agents import FunctionSpanData

        func_data = FunctionSpanData(
            name="get_weather", input='{"city": "NYC"}', output='{"temp": 72}'
        )
        otel_span = processor._start_function_span(func_data, parent_context=None)
        processor._end_function_span(otel_span, func_data, trace_content=False)
        otel_span.end()

        attrs = dict(otel_span.attributes)
        assert GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS not in attrs
        assert GenAIAttributes.GEN_AI_TOOL_CALL_RESULT not in attrs

    def test_none_input_output_omitted(self, tracer_and_exporter, processor):
        """None input/output must not produce attributes."""
        from agents import FunctionSpanData

        func_data = FunctionSpanData(name="noop", input=None, output=None)
        otel_span = processor._start_function_span(func_data, parent_context=None)
        processor._end_function_span(otel_span, func_data, trace_content=True)
        otel_span.end()

        attrs = dict(otel_span.attributes)
        assert GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS not in attrs
        assert GenAIAttributes.GEN_AI_TOOL_CALL_RESULT not in attrs

    def test_non_string_output_coerced(self, tracer_and_exporter, processor):
        """Non-string output must be str()-converted."""
        from agents import FunctionSpanData

        func_data = FunctionSpanData(name="calc", input="2+2", output=4)
        otel_span = processor._start_function_span(func_data, parent_context=None)
        processor._end_function_span(otel_span, func_data, trace_content=True)
        otel_span.end()

        attrs = dict(otel_span.attributes)
        assert attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS] == "2+2"
        assert attrs[GenAIAttributes.GEN_AI_TOOL_CALL_RESULT] == "4"


# ---------------------------------------------------------------------------
# Tests: _start_generation_span
# ---------------------------------------------------------------------------

class TestStartGenerationSpan:
    """Unit tests for the extracted _start_generation_span handler."""

    def test_returns_span_with_chat_attributes(self, tracer_and_exporter, processor):
        """Must return 'openai.response' span with operation_name=chat."""
        otel_span = processor._start_generation_span(parent_context=None)

        assert otel_span is not None
        assert otel_span.name == "openai.response"
        attrs = dict(otel_span.attributes)
        assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"

        otel_span.end()

    def test_span_kind_is_client(self, tracer_and_exporter, processor):
        """Generation/response spans must be CLIENT kind."""
        otel_span = processor._start_generation_span(parent_context=None)

        assert otel_span.kind == SpanKind.CLIENT

        otel_span.end()


# ---------------------------------------------------------------------------
# Tests: _start_realtime_span
# ---------------------------------------------------------------------------

class TestStartRealtimeSpan:
    """Unit tests for the extracted _start_realtime_span handler."""

    def test_speech_span_attributes(self, tracer_and_exporter, processor):
        """Speech span must have correct name and operation."""
        span_data = MagicMock()
        span_data.model = "gpt-4o-realtime-preview"

        otel_span = processor._start_realtime_span(
            span_data, parent_context=None,
            span_name="openai.realtime.speech", operation="speech",
        )

        assert otel_span is not None
        assert otel_span.name == "openai.realtime.speech"
        attrs = dict(otel_span.attributes)
        assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "speech"
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"
        assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-realtime-preview"

        otel_span.end()

    def test_transcription_span_attributes(self, tracer_and_exporter, processor):
        """Transcription span must have correct name and operation."""
        span_data = MagicMock()
        span_data.model = "whisper-1"

        otel_span = processor._start_realtime_span(
            span_data, parent_context=None,
            span_name="openai.realtime.transcription", operation="transcription",
        )

        assert otel_span.name == "openai.realtime.transcription"
        attrs = dict(otel_span.attributes)
        assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "transcription"
        assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "whisper-1"

        otel_span.end()

    def test_speech_group_span_no_model(self, tracer_and_exporter, processor):
        """Speech group span with no model → model attribute omitted."""
        span_data = MagicMock(spec=[])  # no attributes at all

        otel_span = processor._start_realtime_span(
            span_data, parent_context=None,
            span_name="openai.realtime.speech_group", operation="speech_group",
        )

        assert otel_span.name == "openai.realtime.speech_group"
        attrs = dict(otel_span.attributes)
        assert GenAIAttributes.GEN_AI_REQUEST_MODEL not in attrs

        otel_span.end()

    def test_span_kind_is_client(self, tracer_and_exporter, processor):
        """All realtime spans must be CLIENT kind."""
        span_data = MagicMock(spec=[])

        otel_span = processor._start_realtime_span(
            span_data, parent_context=None,
            span_name="openai.realtime.speech", operation="speech",
        )

        assert otel_span.kind == SpanKind.CLIENT

        otel_span.end()


# ---------------------------------------------------------------------------
# Tests: _extract_tool_definitions (pure function)
# ---------------------------------------------------------------------------

class TestExtractToolDefinitions:
    """Unit tests for the extracted _extract_tool_definitions helper."""

    def test_function_wrapped_tool(self):
        """Tool with .function wrapper → {type, function: {name, description, parameters}}."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_tool_definitions,
        )

        func = MagicMock()
        func.name = "get_weather"
        func.description = "Get weather data"
        func.parameters = {"type": "object", "properties": {"city": {"type": "string"}}}

        tool = MagicMock()
        tool.function = func
        tool.type = "function"

        result = _extract_tool_definitions([tool])

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather data"
        assert "properties" in result[0]["function"]["parameters"]

    def test_direct_function_tool(self):
        """Tool with direct .name (no .function wrapper) → {name, description}."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_tool_definitions,
        )

        tool = MagicMock(spec=["name", "description", "parameters"])
        tool.name = "search"
        tool.description = "Search the web"
        tool.parameters = {"type": "object"}

        result = _extract_tool_definitions([tool])

        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["description"] == "Search the web"

    def test_empty_tools_list(self):
        """Empty tools list → empty result."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_tool_definitions,
        )

        result = _extract_tool_definitions([])
        assert result == []

    def test_none_tools(self):
        """None tools → empty result."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_tool_definitions,
        )

        result = _extract_tool_definitions(None)
        assert result == []

    def test_mixed_tool_formats(self):
        """Mix of function-wrapped and direct tools."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_tool_definitions,
        )

        func = MagicMock()
        func.name = "tool_a"
        func.description = "Tool A"
        func.parameters = {}

        wrapped = MagicMock()
        wrapped.function = func
        wrapped.type = "function"

        direct = MagicMock(spec=["name", "description"])
        direct.name = "tool_b"
        direct.description = "Tool B"

        result = _extract_tool_definitions([wrapped, direct])
        assert len(result) == 2
        names = {r.get("name") or r.get("function", {}).get("name") for r in result}
        assert names == {"tool_a", "tool_b"}


# ---------------------------------------------------------------------------
# Tests: _end_generation_span
# ---------------------------------------------------------------------------

class TestEndGenerationSpan:
    """Unit tests for the extracted _end_generation_span method."""

    def test_extracts_prompt_attributes(self, tracer_and_exporter, processor):
        """Must call _extract_prompt_attributes with input data."""
        tracer, exporter = tracer_and_exporter
        otel_span = tracer.start_span("test-gen")

        span_data = MagicMock()
        span_data.input = [{"role": "user", "content": "Hello"}]
        span_data.response = None

        processor._end_generation_span(otel_span, span_data, trace_content=True)

        raw = otel_span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert raw is not None
        messages = json.loads(raw)
        assert messages[0]["role"] == "user"

        otel_span.end()

    def test_extracts_tool_definitions(self, tracer_and_exporter, processor):
        """Must extract and set tool definitions from response.tools."""
        tracer, exporter = tracer_and_exporter
        otel_span = tracer.start_span("test-gen")

        func = MagicMock()
        func.name = "search"
        func.description = "Search"
        func.parameters = {}
        tool = MagicMock()
        tool.function = func
        tool.type = "function"

        response = MagicMock()
        response.tools = [tool]
        response.output = []
        response.model = "gpt-4o"
        response.id = "resp-1"
        response.temperature = None
        response.max_output_tokens = None
        response.top_p = None
        response.frequency_penalty = None
        response.finish_reason = "stop"
        response.usage = None

        span_data = MagicMock()
        span_data.input = []
        span_data.response = response

        processor._end_generation_span(otel_span, span_data, trace_content=True)

        raw = otel_span.attributes.get(GenAIAttributes.GEN_AI_TOOL_DEFINITIONS)
        assert raw is not None
        defs = json.loads(raw)
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "search"

        otel_span.end()

    def test_no_tool_definitions_when_content_gated(self, tracer_and_exporter, processor):
        """Tool definitions must NOT be set when trace_content=False."""
        tracer, exporter = tracer_and_exporter
        otel_span = tracer.start_span("test-gen")

        func = MagicMock()
        func.name = "search"
        func.description = "Search"
        func.parameters = {}
        tool = MagicMock()
        tool.function = func
        tool.type = "function"

        response = MagicMock()
        response.tools = [tool]
        response.output = []
        response.model = "gpt-4o"
        response.id = "resp-1"
        response.temperature = None
        response.max_output_tokens = None
        response.top_p = None
        response.frequency_penalty = None
        response.finish_reason = "stop"
        response.usage = None

        span_data = MagicMock()
        span_data.input = []
        span_data.response = response

        processor._end_generation_span(otel_span, span_data, trace_content=False)

        assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS not in otel_span.attributes

        otel_span.end()

    def test_extracts_response_attributes(self, tracer_and_exporter, processor):
        """Must extract response model, id, etc."""
        tracer, exporter = tracer_and_exporter
        otel_span = tracer.start_span("test-gen")

        content_item = MagicMock()
        content_item.type = "output_text"
        content_item.text = "Hello!"

        output_msg = MagicMock()
        output_msg.content = [content_item]
        output_msg.role = "assistant"
        output_msg.name = None

        response = MagicMock()
        response.tools = []
        response.output = [output_msg]
        response.model = "gpt-4o-mini"
        response.id = "resp-abc"
        response.temperature = 0.7
        response.max_output_tokens = 100
        response.top_p = 1.0
        response.frequency_penalty = None
        response.finish_reason = "stop"
        response.usage = None

        span_data = MagicMock()
        span_data.input = []
        span_data.response = response

        processor._end_generation_span(otel_span, span_data, trace_content=True)

        assert otel_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL) == "gpt-4o-mini"
        assert otel_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID) == "resp-abc"

        otel_span.end()

    def test_no_response_no_crash(self, tracer_and_exporter, processor):
        """span_data.response=None must not raise."""
        tracer, exporter = tracer_and_exporter
        otel_span = tracer.start_span("test-gen")

        span_data = MagicMock()
        span_data.input = []
        span_data.response = None

        # Should not raise
        processor._end_generation_span(otel_span, span_data, trace_content=True)

        otel_span.end()


# ---------------------------------------------------------------------------
# Tests: _set_realtime_io_attributes
# ---------------------------------------------------------------------------

class TestSetRealtimeIOAttributes:
    """Unit tests for the extracted _set_realtime_io_attributes method."""

    def test_speech_span_input_and_output(self, tracer_and_exporter, processor):
        """SpeechSpanData with input text and output text → both messages set."""
        tracer, _ = tracer_and_exporter
        otel_span = tracer.start_span("test-rt")

        span_data = MagicMock()
        span_data.input = "What is the weather?"
        span_data.output = "It's sunny."

        processor._set_realtime_io_attributes(otel_span, span_data, has_output=True)

        raw_in = otel_span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert raw_in is not None
        in_msgs = json.loads(raw_in)
        assert in_msgs[0]["role"] == "user"
        assert in_msgs[0]["parts"][0]["content"] == "What is the weather?"

        raw_out = otel_span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        assert raw_out is not None
        out_msgs = json.loads(raw_out)
        assert out_msgs[0]["role"] == "assistant"
        assert out_msgs[0]["parts"][0]["content"] == "It's sunny."

        otel_span.end()

    def test_transcription_span_input_and_output(self, tracer_and_exporter, processor):
        """TranscriptionSpanData with audio input (non-binary) and text output."""
        tracer, _ = tracer_and_exporter
        otel_span = tracer.start_span("test-rt")

        span_data = MagicMock()
        span_data.input = "audio-description-text"
        span_data.output = "Transcribed text here"

        processor._set_realtime_io_attributes(otel_span, span_data, has_output=True)

        raw_in = otel_span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert raw_in is not None

        raw_out = otel_span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        assert raw_out is not None

        otel_span.end()

    def test_binary_input_skipped(self, tracer_and_exporter, processor):
        """Binary input (bytes/bytearray) must NOT be set as input message."""
        tracer, _ = tracer_and_exporter
        otel_span = tracer.start_span("test-rt")

        span_data = MagicMock()
        span_data.input = b"\x00\x01\x02"
        span_data.output = "Transcribed"

        processor._set_realtime_io_attributes(otel_span, span_data, has_output=True)

        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in otel_span.attributes

        otel_span.end()

    def test_binary_output_skipped(self, tracer_and_exporter, processor):
        """Binary output (bytes/bytearray) must NOT be set as output message."""
        tracer, _ = tracer_and_exporter
        otel_span = tracer.start_span("test-rt")

        span_data = MagicMock()
        span_data.input = "Hello"
        span_data.output = b"\x00\x01\x02"

        processor._set_realtime_io_attributes(otel_span, span_data, has_output=True)

        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in otel_span.attributes

        otel_span.end()

    def test_speech_group_no_output(self, tracer_and_exporter, processor):
        """SpeechGroupSpanData with has_output=False → only input set."""
        tracer, _ = tracer_and_exporter
        otel_span = tracer.start_span("test-rt")

        span_data = MagicMock()
        span_data.input = "Group input"
        span_data.output = None

        processor._set_realtime_io_attributes(otel_span, span_data, has_output=False)

        raw_in = otel_span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert raw_in is not None
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in otel_span.attributes

        otel_span.end()

    def test_none_input_skipped(self, tracer_and_exporter, processor):
        """None input → no input message attribute."""
        tracer, _ = tracer_and_exporter
        otel_span = tracer.start_span("test-rt")

        span_data = MagicMock()
        span_data.input = None
        span_data.output = "Output text"

        processor._set_realtime_io_attributes(otel_span, span_data, has_output=True)

        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in otel_span.attributes

        otel_span.end()

    def test_output_has_finish_reason_empty(self, tracer_and_exporter, processor):
        """Realtime output messages must include finish_reason: '' (empty string)."""
        tracer, _ = tracer_and_exporter
        otel_span = tracer.start_span("test-rt")

        span_data = MagicMock()
        span_data.input = None
        span_data.output = "Some output"

        processor._set_realtime_io_attributes(otel_span, span_data, has_output=True)

        raw_out = otel_span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        out_msgs = json.loads(raw_out)
        assert out_msgs[0]["finish_reason"] == ""

        otel_span.end()


# ---------------------------------------------------------------------------
# Integration: on_span_start/on_span_end still work end-to-end
# (These confirm refactoring doesn't break the public API)
# ---------------------------------------------------------------------------

class TestOnSpanStartEndToEnd:
    """Verify on_span_start dispatches correctly after refactoring."""

    def _run_span(self, processor, exporter, span_data, trace_id="e2e-trace"):
        mock_trace = MagicMock()
        mock_trace.trace_id = trace_id
        processor.on_trace_start(mock_trace)

        span = MockAgentSpan(span_data, trace_id=trace_id)
        processor.on_span_start(span)
        processor.on_span_end(span)
        processor.on_trace_end(mock_trace)

        return exporter.get_finished_spans()

    def test_agent_span_created(self, tracer_and_exporter, processor):
        from agents import AgentSpanData
        _, exporter = tracer_and_exporter

        spans = self._run_span(
            processor, exporter,
            AgentSpanData(name="TestAgent", handoffs=[], tools=[], output_type=""),
        )
        names = [s.name for s in spans]
        assert "TestAgent.agent" in names

    def test_handoff_span_created(self, tracer_and_exporter, processor):
        from agents import HandoffSpanData
        _, exporter = tracer_and_exporter

        spans = self._run_span(
            processor, exporter,
            HandoffSpanData(from_agent="A", to_agent="B"),
        )
        names = [s.name for s in spans]
        assert any("handoff" in n for n in names)

    def test_function_span_created(self, tracer_and_exporter, processor):
        from agents import FunctionSpanData
        _, exporter = tracer_and_exporter

        spans = self._run_span(
            processor, exporter,
            FunctionSpanData(name="my_tool", input="", output=""),
        )
        names = [s.name for s in spans]
        assert "my_tool.tool" in names

    def test_generation_span_created(self, tracer_and_exporter, processor):
        from agents import GenerationSpanData
        _, exporter = tracer_and_exporter

        spans = self._run_span(
            processor, exporter,
            GenerationSpanData(model="gpt-4o", model_config={}),
        )
        names = [s.name for s in spans]
        assert "openai.response" in names

    def test_error_status_propagated(self, tracer_and_exporter, processor):
        from agents import FunctionSpanData
        _, exporter = tracer_and_exporter

        mock_trace = MagicMock()
        mock_trace.trace_id = "err-trace"
        processor.on_trace_start(mock_trace)

        span_data = FunctionSpanData(name="fail_tool", input="", output="")
        span = MockAgentSpan(span_data, trace_id="err-trace", error=RuntimeError("boom"))
        processor.on_span_start(span)
        processor.on_span_end(span)
        processor.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        tool_span = next(s for s in spans if s.name == "fail_tool.tool")
        assert tool_span.status.status_code.name == "ERROR"
