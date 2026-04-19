"""
Comprehensive OTel GenAI semconv compliance tests for openai-agents instrumentation.

Tests validate that all message formatting, attribute names, and values conform to
the OTel GenAI semantic conventions (parts-based schema, v1.40.0+).

Reference schemas: semconv-schemas/gen-ai-input-messages.json, gen-ai-output-messages.json
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracer_and_exporter():
    """Create a tracer provider with in-memory exporter for unit tests."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


@pytest.fixture
def processor(tracer_and_exporter):
    """Create an OpenTelemetryTracingProcessor with a fresh tracer."""
    from opentelemetry.instrumentation.openai_agents._hooks import (
        OpenTelemetryTracingProcessor,
    )

    tracer, _ = tracer_and_exporter
    return OpenTelemetryTracingProcessor(tracer)


# ---------------------------------------------------------------------------
# Helper: mock span data objects
# ---------------------------------------------------------------------------

class MockAgentSpan:
    def __init__(self, span_data, trace_id="test-trace", error=None):
        self.span_data = span_data
        self.trace_id = trace_id
        self.error = error


class MockGenerationSpanData:
    """Mock for agents.GenerationSpanData."""

    def __init__(self, input=None, response=None):
        self.input = input or []
        self.response = response


class ResponseSpanData:
    """Lightweight stub whose __name__ is 'ResponseSpanData' (no MagicMock mutation)."""

    def __init__(self, input=None, response=None):
        self.input = input or []
        self.response = response


class MockResponseOutput:
    """Mock for a response output item with text content."""

    def __init__(self, role="assistant", content=None, text=None, name=None,
                 call_id=None, arguments=None, type=None):
        self.role = role
        self.content = content
        self.text = text
        self.name = name
        self.call_id = call_id
        self.arguments = arguments
        if type is None and content is not None:
            self.type = "message"
        elif type is None and call_id is not None:
            self.type = "function_call"
        else:
            self.type = type


class MockContentItem:
    """Mock for a content item inside ResponseOutputMessage."""

    def __init__(self, text=None):
        self.text = text


class MockUsage:
    def __init__(self, input_tokens=10, output_tokens=20, total_tokens=30):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.prompt_tokens = None
        self.completion_tokens = None


class MockResponse:
    """Mock for the response object from GenerationSpanData."""

    def __init__(self, output=None, model=None, temperature=None,
                 max_output_tokens=None, top_p=None, frequency_penalty=None,
                 usage=None, finish_reason=None, id=None, tools=None):
        self.output = output or []
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.usage = usage
        self.finish_reason = finish_reason
        self.id = id
        self.tools = tools or []


class MockFunction:
    """Mock for a tool function definition."""

    def __init__(self, name="", description="", parameters=None):
        self.name = name
        self.description = description
        self.parameters = parameters


class MockTool:
    """Mock for a tool definition with function wrapper."""

    def __init__(self, function=None, type="function"):
        self.function = function
        self.type = type


# ---------------------------------------------------------------------------
# P1-1: gen_ai.provider.name replaces gen_ai.system
# ---------------------------------------------------------------------------

class TestProviderName:
    """Verify gen_ai.provider.name is used instead of deprecated gen_ai.system."""

    def test_generation_span_uses_provider_name(self, tracer_and_exporter):
        """GenerationSpanData spans must use gen_ai.provider.name, not gen_ai.system."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )
        from agents import GenerationSpanData

        tracer, exporter = tracer_and_exporter
        proc = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-pn-1"
        proc.on_trace_start(mock_trace)

        gen_data = GenerationSpanData(model="gpt-4o", model_config={})
        span = MockAgentSpan(gen_data, trace_id="test-pn-1")

        proc.on_span_start(span)
        proc.on_span_end(span)
        proc.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        response_span = next((s for s in spans if s.name == "openai.response"), None)
        assert response_span is not None, "Expected openai.response span"

        attrs = dict(response_span.attributes)
        assert GenAIAttributes.GEN_AI_PROVIDER_NAME in attrs, (
            f"Expected gen_ai.provider.name attribute, got keys: {list(attrs.keys())}"
        )
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"

    def test_agent_span_uses_provider_name_openai(self, tracer_and_exporter):
        """Agent spans must use gen_ai.provider.name = 'openai', NOT 'openai_agents'."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )
        from agents import AgentSpanData

        tracer, exporter = tracer_and_exporter
        proc = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-pn-2"
        proc.on_trace_start(mock_trace)

        agent_data = AgentSpanData(name="TestAgent", handoffs=[], tools=[], output_type="")
        span = MockAgentSpan(agent_data, trace_id="test-pn-2")

        proc.on_span_start(span)
        proc.on_span_end(span)
        proc.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        agent_span = next((s for s in spans if s.name == "TestAgent.agent"), None)
        assert agent_span is not None, "Expected TestAgent.agent span"

        attrs = dict(agent_span.attributes)
        assert GenAIAttributes.GEN_AI_PROVIDER_NAME in attrs
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai", (
            f"Agent span provider name should be 'openai', got '{attrs.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)}'"
        )

    def test_workflow_span_uses_provider_name(self, tracer_and_exporter):
        """Workflow spans must use gen_ai.provider.name."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )

        tracer, exporter = tracer_and_exporter
        proc = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-pn-3"
        proc.on_trace_start(mock_trace)
        proc.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        wf_span = next((s for s in spans if s.name == "Agent Workflow"), None)
        assert wf_span is not None

        attrs = dict(wf_span.attributes)
        assert GenAIAttributes.GEN_AI_PROVIDER_NAME in attrs
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"

    def test_tool_span_uses_provider_name(self, tracer_and_exporter):
        """Tool spans must use gen_ai.provider.name."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )
        from agents import FunctionSpanData

        tracer, exporter = tracer_and_exporter
        proc = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-pn-4"
        proc.on_trace_start(mock_trace)

        func_data = FunctionSpanData(name="get_weather", input="", output="")
        span = MockAgentSpan(func_data, trace_id="test-pn-4")

        proc.on_span_start(span)
        proc.on_span_end(span)
        proc.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        tool_span = next((s for s in spans if s.name == "get_weather.tool"), None)
        assert tool_span is not None

        attrs = dict(tool_span.attributes)
        assert GenAIAttributes.GEN_AI_PROVIDER_NAME in attrs
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"

    def test_handoff_span_uses_provider_name(self, tracer_and_exporter):
        """Handoff spans must use gen_ai.provider.name."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )
        from agents import HandoffSpanData

        tracer, exporter = tracer_and_exporter
        proc = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-pn-5"
        proc.on_trace_start(mock_trace)

        handoff_data = HandoffSpanData(from_agent="AgentA", to_agent="AgentB")
        span = MockAgentSpan(handoff_data, trace_id="test-pn-5")

        proc.on_span_start(span)
        proc.on_span_end(span)
        proc.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        handoff_span = next((s for s in spans if "handoff" in s.name), None)
        assert handoff_span is not None

        attrs = dict(handoff_span.attributes)
        assert GenAIAttributes.GEN_AI_PROVIDER_NAME in attrs
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"


# ---------------------------------------------------------------------------
# P1-2 / P1-3: Input & Output messages use parts-based schema
# ---------------------------------------------------------------------------

class TestInputMessagePartsFormat:
    """Verify gen_ai.input.messages uses {role, parts} schema."""

    def test_text_message_has_parts(self, tracer_and_exporter):
        """Simple text message must have parts: [{type: 'text', content: '...'}]."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, exporter = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{"role": "user", "content": "Hello world"}]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert raw is not None, "gen_ai.input.messages should be set"
        messages = json.loads(raw)

        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "user"
        assert "parts" in msg, f"Message must have 'parts' key, got keys: {list(msg.keys())}"
        assert "content" not in msg, "Top-level 'content' key should NOT be present (use parts instead)"

        parts = msg["parts"]
        assert len(parts) == 1
        assert parts[0]["type"] == "text"
        assert parts[0]["content"] == "Hello world"

        span.end()

    def test_tool_call_message_has_parts(self, tracer_and_exporter):
        """Assistant tool call message must use parts with type 'tool_call'."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_123",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}'
                }
            }]
        }]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)
        msg = messages[0]

        assert msg["role"] == "assistant"
        assert "parts" in msg
        assert "tool_calls" not in msg, "Top-level 'tool_calls' must NOT be present (use parts)"

        tool_part = msg["parts"][0]
        assert tool_part["type"] == "tool_call"
        assert tool_part["id"] == "call_123"
        assert tool_part["name"] == "get_weather"
        # Arguments must be parsed object, not string
        assert isinstance(tool_part["arguments"], dict), (
            f"arguments must be dict (parsed object), got {type(tool_part['arguments'])}"
        )
        assert tool_part["arguments"] == {"city": "NYC"}

        span.end()

    def test_tool_result_message_has_parts(self, tracer_and_exporter):
        """Tool result message must use parts with type 'tool_call_response'."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "72°F, sunny"
        }]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)
        msg = messages[0]

        assert msg["role"] == "tool"
        assert "parts" in msg
        assert "content" not in msg, "Top-level 'content' must NOT be present for tool messages"
        assert "tool_call_id" not in msg, "Top-level 'tool_call_id' must NOT be present"

        tool_part = msg["parts"][0]
        assert tool_part["type"] == "tool_call_response"
        assert tool_part["id"] == "call_123"
        assert tool_part["response"] == "72°F, sunny"

        span.end()

    def test_agents_sdk_function_call_format(self, tracer_and_exporter):
        """Agents SDK function_call type messages must convert to tool_call parts."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        # Agents SDK format uses dict with 'type' key, no 'role'
        input_data = [{
            "type": "function_call",
            "id": "fc_1",
            "name": "search",
            "arguments": '{"q": "test"}',
        }]

        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)
        msg = messages[0]

        assert msg["role"] == "assistant"
        assert "parts" in msg
        tool_part = msg["parts"][0]
        assert tool_part["type"] == "tool_call"
        assert tool_part["name"] == "search"
        assert isinstance(tool_part["arguments"], dict)
        assert tool_part["arguments"]["q"] == "test"

        span.end()

    def test_agents_sdk_function_call_output_format(self, tracer_and_exporter):
        """Agents SDK function_call_output type must convert to tool_call_response parts."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        # Agents SDK format uses dict with 'type' key, no 'role'
        input_data = [{
            "type": "function_call_output",
            "call_id": "fc_1",
            "output": "Result data",
        }]

        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)
        msg = messages[0]

        assert msg["role"] == "tool"
        assert "parts" in msg
        tool_resp_part = msg["parts"][0]
        assert tool_resp_part["type"] == "tool_call_response"
        assert tool_resp_part["id"] == "fc_1"
        assert tool_resp_part["response"] == "Result data"

        span.end()

    def test_list_content_with_tool_calls_preserves_structure(self, tracer_and_exporter):
        """List content + tool_calls must preserve structured parts, not stringify."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ],
            "tool_calls": [{
                "id": "call_1",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}'
                }
            }]
        }]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert raw is not None
        messages = json.loads(raw)
        msg = messages[0]

        assert msg["role"] == "assistant"
        assert "parts" in msg

        parts = msg["parts"]
        # Expect: text part, uri part (image), tool_call part
        assert len(parts) == 3, f"Expected 3 parts (text + image + tool_call), got {len(parts)}: {parts}"

        text_part = parts[0]
        assert text_part["type"] == "text"
        assert text_part["content"] == "Let me check"

        image_part = parts[1]
        assert image_part["type"] == "uri", (
            f"image_url must map to 'uri' part, got type '{image_part['type']}'"
        )
        assert image_part["modality"] == "image"
        assert image_part["uri"] == "https://example.com/img.png"

        tool_part = parts[2]
        assert tool_part["type"] == "tool_call"
        assert tool_part["name"] == "get_weather"

        span.end()

    def test_string_content_with_tool_calls(self, tracer_and_exporter):
        """String content + tool_calls should produce text part + tool_call part."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{
            "role": "assistant",
            "content": "Let me look that up",
            "tool_calls": [{
                "id": "call_2",
                "function": {
                    "name": "search",
                    "arguments": '{"q": "test"}'
                }
            }]
        }]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert raw is not None
        messages = json.loads(raw)
        msg = messages[0]

        parts = msg["parts"]
        assert len(parts) == 2, f"Expected 2 parts (text + tool_call), got {len(parts)}"
        assert parts[0]["type"] == "text"
        assert parts[0]["content"] == "Let me look that up"
        assert parts[1]["type"] == "tool_call"

        span.end()

    def test_none_content_message(self, tracer_and_exporter):
        """Messages with None content should still produce valid parts."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{"role": "assistant", "content": None}]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        if raw:
            messages = json.loads(raw)
            if messages:
                msg = messages[0]
                assert "parts" in msg

        span.end()

    def test_empty_input_data(self, tracer_and_exporter):
        """Empty input data should not set the attribute."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        _extract_prompt_attributes(span, [], trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert raw is None, "Empty input should not set gen_ai.input.messages"

        span.end()


class TestOutputMessagePartsFormat:
    """Verify gen_ai.output.messages uses {role, parts, finish_reason} schema."""

    def test_text_output_has_parts(self, tracer_and_exporter):
        """Text output must be wrapped in parts: [{type: 'text', content: '...'}]."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        content_item = MockContentItem(text="Hello!")
        output_item = MockResponseOutput(role="assistant", content=[content_item])
        response = MockResponse(
            output=[output_item],
            model="gpt-4o",
            usage=MockUsage(),
            finish_reason="stop",
            id="resp_123",
        )

        _extract_response_attributes(span, response, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        assert raw is not None
        messages = json.loads(raw)

        msg = messages[0]
        assert msg["role"] == "assistant"
        assert "parts" in msg, f"Output message must have 'parts', got keys: {list(msg.keys())}"
        assert "content" not in msg, "Top-level 'content' must NOT be present"

        parts = msg["parts"]
        assert len(parts) >= 1
        assert parts[0]["type"] == "text"
        assert parts[0]["content"] == "Hello!"

        span.end()

    def test_tool_call_output_has_parts(self, tracer_and_exporter):
        """Tool call output must use parts with type 'tool_call'."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        output_item = MockResponseOutput(
            name="get_weather", call_id="call_456", arguments='{"city": "London"}'
        )
        response = MockResponse(
            output=[output_item],
            model="gpt-4o",
            usage=MockUsage(),
        )

        _extract_response_attributes(span, response, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        assert raw is not None
        messages = json.loads(raw)

        msg = messages[0]
        assert msg["role"] == "assistant"
        assert "parts" in msg
        assert "tool_calls" not in msg, "Top-level 'tool_calls' must NOT be present"

        tool_part = msg["parts"][0]
        assert tool_part["type"] == "tool_call"
        assert tool_part["name"] == "get_weather"
        assert tool_part["id"] == "call_456"

        span.end()

    def test_output_finish_reason_present(self, tracer_and_exporter):
        """Output messages must have finish_reason at message level."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        content_item = MockContentItem(text="Done")
        output_item = MockResponseOutput(role="assistant", content=[content_item])
        response = MockResponse(
            output=[output_item],
            model="gpt-4o",
            usage=MockUsage(),
            finish_reason="stop",
        )

        _extract_response_attributes(span, response, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        messages = json.loads(raw)

        msg = messages[0]
        assert "finish_reason" in msg, "finish_reason is required per schema"
        assert msg["finish_reason"] == "stop"

        span.end()

    def test_output_finish_reason_empty_when_unknown(self, tracer_and_exporter):
        """finish_reason must be '' (not fabricated 'stop') when unknown."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        content_item = MockContentItem(text="Done")
        output_item = MockResponseOutput(role="assistant", content=[content_item])
        response = MockResponse(
            output=[output_item],
            model="gpt-4o",
            usage=MockUsage(),
            finish_reason=None,
        )

        _extract_response_attributes(span, response, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        messages = json.loads(raw)

        msg = messages[0]
        assert "finish_reason" in msg, "finish_reason must always be present (required by schema)"
        # When finish_reason is unknown, it should be empty string, NOT fabricated "stop"
        assert msg["finish_reason"] == "", (
            f"finish_reason should be '' when unknown, got '{msg['finish_reason']}'"
        )

        span.end()

    def test_message_with_empty_content_and_name_not_tool_call(self, tracer_and_exporter):
        """ResponseOutputMessage with empty content + participant name must not become a tool call.

        Semconv: ToolCallRequestPart.name MUST identify a tool, not a participant.
        """
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )
        from types import SimpleNamespace

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        output = SimpleNamespace(
            type="message", content=[], name="CustomerServiceBot", role="assistant",
        )
        response = SimpleNamespace(
            temperature=None, max_output_tokens=None, top_p=None,
            model=None, id=None, frequency_penalty=None,
            finish_reason=None, status="completed",
            output=[output], usage=None,
        )

        _extract_response_attributes(span, response, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        if raw:
            messages = json.loads(raw)
            for msg in messages:
                for part in msg.get("parts", []):
                    assert part.get("type") != "tool_call", (
                        "Participant name was misclassified as tool call"
                    )

        span.end()


# ---------------------------------------------------------------------------
# P1-4: Arguments parsed as objects
# ---------------------------------------------------------------------------

class TestArgumentsParsing:
    """Verify tool call arguments are parsed to objects, not kept as strings."""

    def test_string_arguments_parsed_to_dict(self, tracer_and_exporter):
        """JSON string arguments must be parsed to dict."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "weather", "limit": 5}'
                }
            }]
        }]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)
        tool_part = messages[0]["parts"][0]

        assert isinstance(tool_part["arguments"], dict), (
            f"Arguments should be parsed to dict, got {type(tool_part['arguments'])}"
        )
        assert tool_part["arguments"]["query"] == "weather"
        assert tool_part["arguments"]["limit"] == 5

        span.end()

    def test_dict_arguments_kept_as_dict(self, tracer_and_exporter):
        """Dict arguments should stay as dict."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "function": {
                    "name": "search",
                    "arguments": {"query": "test"}
                }
            }]
        }]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)
        tool_part = messages[0]["parts"][0]

        assert isinstance(tool_part["arguments"], dict)
        assert tool_part["arguments"]["query"] == "test"

        span.end()

    def test_invalid_json_arguments_fallback(self, tracer_and_exporter):
        """Invalid JSON string arguments should have best-effort fallback."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "function": {
                    "name": "search",
                    "arguments": "not valid json {"
                }
            }]
        }]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)
        tool_part = messages[0]["parts"][0]

        # Should not crash, arguments should be present in some form
        assert "arguments" in tool_part

        span.end()


# ---------------------------------------------------------------------------
# P1-5 / P1-6: Finish reasons
# ---------------------------------------------------------------------------

class TestFinishReasons:
    """Verify finish reason mapping and top-level attribute."""

    def test_finish_reasons_top_level_attribute(self, tracer_and_exporter):
        """gen_ai.response.finish_reasons must be set as top-level span array."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        content_item = MockContentItem(text="Done")
        output_item = MockResponseOutput(role="assistant", content=[content_item])
        response = MockResponse(
            output=[output_item],
            model="gpt-4o",
            usage=MockUsage(),
            finish_reason="stop",
        )

        _extract_response_attributes(span, response, trace_content=True)

        finish_reasons = span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert finish_reasons is not None, (
            "gen_ai.response.finish_reasons must be set as top-level span attribute"
        )
        assert isinstance(finish_reasons, (list, tuple))
        assert "stop" in finish_reasons

        span.end()

    def test_finish_reasons_tool_calls_mapped_to_singular(self, tracer_and_exporter):
        """OpenAI 'tool_calls' (plural) must map to 'tool_call' (singular)."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        output_item = MockResponseOutput(name="search", call_id="c1", arguments="{}")
        response = MockResponse(
            output=[output_item],
            model="gpt-4o",
            usage=MockUsage(),
            finish_reason="tool_calls",
        )

        _extract_response_attributes(span, response, trace_content=True)

        finish_reasons = span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert finish_reasons is not None

        # Must be singular "tool_call", not plural "tool_calls"
        assert "tool_call" in finish_reasons, (
            f"Expected 'tool_call' (singular), got {finish_reasons}"
        )
        assert "tool_calls" not in finish_reasons

        span.end()

    def test_finish_reasons_none_omits_attribute(self, tracer_and_exporter):
        """When finish_reason is None, top-level attr should be omitted (not fabricated)."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        content_item = MockContentItem(text="Done")
        output_item = MockResponseOutput(role="assistant", content=[content_item])
        response = MockResponse(
            output=[output_item],
            model="gpt-4o",
            usage=MockUsage(),
            finish_reason=None,
        )

        _extract_response_attributes(span, response, trace_content=True)

        finish_reasons = span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        # When None, attribute should be omitted entirely
        if finish_reasons is not None:
            assert "stop" not in finish_reasons, "Must NOT fabricate 'stop' when finish_reason is None"

        span.end()

    def test_finish_reasons_set_without_prompts(self, tracer_and_exporter):
        """finish_reasons must be set even when should_send_prompts() is False."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        content_item = MockContentItem(text="Done")
        output_item = MockResponseOutput(role="assistant", content=[content_item])
        response = MockResponse(
            output=[output_item],
            model="gpt-4o",
            usage=MockUsage(),
            finish_reason="stop",
        )

        # trace_content=False simulates should_send_prompts() returning False
        _extract_response_attributes(span, response, trace_content=False)

        finish_reasons = span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert finish_reasons is not None, (
            "gen_ai.response.finish_reasons must be set even when content tracing is disabled"
        )

        span.end()

    def test_tool_call_top_level_matches_per_message(self, tracer_and_exporter):
        """Top-level finish_reasons must say 'tool_call' when output contains tool calls.

        Semconv: gen_ai.response.finish_reasons corresponds to each generation.
        If the model stopped to emit a tool call, both levels must agree.
        """
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )
        from types import SimpleNamespace

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        tool_output = SimpleNamespace(
            type="function_call",
            content=None,
            name="get_weather",
            arguments='{"city": "London"}',
            call_id="call_123",
        )
        response = SimpleNamespace(
            temperature=None, max_output_tokens=None, top_p=None,
            model=None, id=None, frequency_penalty=None,
            finish_reason=None, status="completed",
            output=[tool_output], usage=None,
        )

        _extract_response_attributes(span, response, trace_content=True)

        finish_reasons = span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert finish_reasons is not None, "Missing gen_ai.response.finish_reasons"
        assert "tool_call" in finish_reasons, (
            f"Expected 'tool_call' in finish_reasons, got {finish_reasons}"
        )

        span.end()


# ---------------------------------------------------------------------------
# P1-7: Operation name
# ---------------------------------------------------------------------------

class TestOperationName:
    """Verify gen_ai.operation.name uses well-known OTel values."""

    def test_generation_span_operation_name_is_chat(self, tracer_and_exporter):
        """GenerationSpanData must use operation name 'chat'."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )
        from agents import GenerationSpanData

        tracer, exporter = tracer_and_exporter
        proc = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-op-1"
        proc.on_trace_start(mock_trace)

        gen_data = GenerationSpanData(model="gpt-4o", model_config={})
        span = MockAgentSpan(gen_data, trace_id="test-op-1")

        proc.on_span_start(span)
        proc.on_span_end(span)
        proc.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        resp_span = next((s for s in spans if s.name == "openai.response"), None)
        assert resp_span is not None

        assert resp_span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat", (
            f"Expected 'chat', got '{resp_span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]}'"
        )

    def test_response_span_data_operation_name_is_chat(self, tracer_and_exporter):
        """ResponseSpanData must use operation name 'chat', NOT 'response'."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )

        tracer, exporter = tracer_and_exporter
        proc = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-op-2"
        proc.on_trace_start(mock_trace)

        # Create a lightweight ResponseSpanData stub (avoids mutating MagicMock.__name__)
        response_data = ResponseSpanData(input=[], response=None)

        span = MockAgentSpan(response_data, trace_id="test-op-2")

        proc.on_span_start(span)
        proc.on_span_end(span)
        proc.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        resp_span = next((s for s in spans if s.name == "openai.response"), None)
        assert resp_span is not None

        op_name = resp_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        assert op_name == "chat", f"Expected 'chat', got '{op_name}'"


# ---------------------------------------------------------------------------
# P2-1 / P2-2: Response model and ID
# ---------------------------------------------------------------------------

class TestResponseAttributes:
    """Verify recommended response attributes are set."""

    def test_response_model_set(self, tracer_and_exporter):
        """gen_ai.response.model should be set from response."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        response = MockResponse(
            model="gpt-4o-2024-08-06",
            usage=MockUsage(),
        )

        _extract_response_attributes(span, response, trace_content=True)

        assert GenAIAttributes.GEN_AI_RESPONSE_MODEL in span.attributes, (
            "gen_ai.response.model should be set"
        )
        assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-4o-2024-08-06"

        span.end()

    def test_response_id_set(self, tracer_and_exporter):
        """gen_ai.response.id should be set from response."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        response = MockResponse(
            model="gpt-4o",
            usage=MockUsage(),
            id="resp_abc123",
        )

        _extract_response_attributes(span, response, trace_content=True)

        assert GenAIAttributes.GEN_AI_RESPONSE_ID in span.attributes, (
            "gen_ai.response.id should be set"
        )
        assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == "resp_abc123"

        span.end()

    def test_frequency_penalty_set_on_span(self, tracer_and_exporter):
        """gen_ai.request.frequency_penalty should be set as span attribute."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        response = MockResponse(
            model="gpt-4o",
            frequency_penalty=0.5,
            usage=MockUsage(),
        )

        _extract_response_attributes(span, response, trace_content=True)

        assert GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY in span.attributes, (
            "gen_ai.request.frequency_penalty should be set on span"
        )
        assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY] == 0.5

        span.end()

    def test_response_model_does_not_overwrite_request_model(self, tracer_and_exporter):
        """response.model must only set gen_ai.response.model, not gen_ai.request.model.

        Semconv: gen_ai.request.model (alias, e.g. 'gpt-4o') and
        gen_ai.response.model (served, e.g. 'gpt-4o-2024-08-06') are distinct.
        """
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )
        from types import SimpleNamespace

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")
        span.set_attribute(GenAIAttributes.GEN_AI_REQUEST_MODEL, "gpt-4o")

        response = SimpleNamespace(
            temperature=None, max_output_tokens=None, top_p=None,
            model="gpt-4o-2024-08-06", id=None, frequency_penalty=None,
            finish_reason=None, status="completed", output=[], usage=None,
        )

        _extract_response_attributes(span, response, trace_content=True)

        assert span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "gpt-4o", (
            "response.model must not overwrite gen_ai.request.model"
        )
        assert span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL) == "gpt-4o-2024-08-06"

        span.end()


# ---------------------------------------------------------------------------
# P2-7: Tool definitions preserve full format
# ---------------------------------------------------------------------------

class TestToolDefinitions:
    """Verify tool definitions preserve the source system's representation."""

    def test_tool_definitions_preserve_type_wrapper(self, tracer_and_exporter):
        """Tool definitions should preserve the 'type: function' wrapper."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )
        from agents import GenerationSpanData

        tracer, exporter = tracer_and_exporter
        proc = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-td-1"
        proc.on_trace_start(mock_trace)

        gen_data = GenerationSpanData(model="gpt-4o", model_config={})
        gen_data.input = []

        # Create response with tools
        func = MockFunction(name="search", description="Search for data", parameters={"type": "object"})
        tool = MockTool(function=func, type="function")
        gen_data.response = MockResponse(
            model="gpt-4o",
            tools=[tool],
            usage=MockUsage(),
        )

        span = MockAgentSpan(gen_data, trace_id="test-td-1")

        proc.on_span_start(span)
        proc.on_span_end(span)
        proc.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        resp_span = next((s for s in spans if s.name == "openai.response"), None)
        assert resp_span is not None

        raw_defs = resp_span.attributes.get(GenAIAttributes.GEN_AI_TOOL_DEFINITIONS)
        assert raw_defs is not None, "gen_ai.tool.definitions must be set when tools are present"
        defs = json.loads(raw_defs)
        assert len(defs) >= 1
        tool_def = defs[0]
        # Per spec: preserve source system's representation
        assert "type" in tool_def, "Tool definition should preserve 'type' field"
        assert tool_def["type"] == "function"
        assert "function" in tool_def, "Tool definition should preserve 'function' wrapper"


# ---------------------------------------------------------------------------
# P2-5: Realtime messages parts format
# ---------------------------------------------------------------------------

class TestRealtimeMessageFormat:
    """Verify realtime LLM span messages use parts-based format."""

    def test_realtime_llm_span_input_uses_parts(self):
        """Realtime input messages must use parts-based format."""
        from opentelemetry.instrumentation.openai_agents._realtime_wrappers import (
            RealtimeTracingState,
        )

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        state = RealtimeTracingState(tracer)
        state.start_workflow_span("TestAgent")
        state.start_agent_span("TestAgent")

        state.record_prompt("user", "What is the weather?")

        with patch(
            "opentelemetry.instrumentation.openai_agents._realtime_wrappers.should_send_prompts",
            return_value=True,
        ):
            state.create_llm_span("It's sunny!")

        state.cleanup()
        state.end_workflow_span()

        spans = exporter.get_finished_spans()
        llm_span = next((s for s in spans if s.name == "openai.realtime"), None)
        assert llm_span is not None

        raw_input = llm_span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert raw_input is not None, "gen_ai.input.messages must be set on realtime LLM span"
        messages = json.loads(raw_input)
        msg = messages[0]
        assert "parts" in msg, f"Realtime input must use parts format, got keys: {list(msg.keys())}"

    def test_realtime_llm_span_output_uses_parts(self):
        """Realtime output messages must use parts-based format."""
        from opentelemetry.instrumentation.openai_agents._realtime_wrappers import (
            RealtimeTracingState,
        )

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        state = RealtimeTracingState(tracer)
        state.start_workflow_span("TestAgent")
        state.start_agent_span("TestAgent")

        state.record_prompt("user", "Hello")

        with patch(
            "opentelemetry.instrumentation.openai_agents._realtime_wrappers.should_send_prompts",
            return_value=True,
        ):
            state.create_llm_span("Hi there!")

        state.cleanup()
        state.end_workflow_span()

        spans = exporter.get_finished_spans()
        llm_span = next((s for s in spans if s.name == "openai.realtime"), None)
        assert llm_span is not None

        raw_output = llm_span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        assert raw_output is not None, "gen_ai.output.messages must be set on realtime LLM span"
        messages = json.loads(raw_output)
        msg = messages[0]
        assert "parts" in msg, f"Realtime output must use parts format, got keys: {list(msg.keys())}"

    def test_realtime_does_not_fabricate_stop(self):
        """Realtime must NOT fabricate finish_reason 'stop'."""
        from opentelemetry.instrumentation.openai_agents._realtime_wrappers import (
            RealtimeTracingState,
        )

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        state = RealtimeTracingState(tracer)
        state.start_workflow_span("TestAgent")
        state.start_agent_span("TestAgent")

        state.record_prompt("user", "Test")

        with patch(
            "opentelemetry.instrumentation.openai_agents._realtime_wrappers.should_send_prompts",
            return_value=True,
        ):
            state.create_llm_span("Response")

        state.cleanup()
        state.end_workflow_span()

        spans = exporter.get_finished_spans()
        llm_span = next((s for s in spans if s.name == "openai.realtime"), None)
        assert llm_span is not None

        raw_output = llm_span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        assert raw_output is not None, "gen_ai.output.messages must be set on realtime LLM span"
        messages = json.loads(raw_output)
        msg = messages[0]
        # finish_reason should be empty string, not fabricated "stop"
        fr = msg.get("finish_reason")
        assert fr == "", (
            f"Realtime should not fabricate finish_reason, got '{fr}'"
        )


# ---------------------------------------------------------------------------
# Realtime operation name
# ---------------------------------------------------------------------------

class TestRealtimeOperationName:
    """Verify realtime spans set gen_ai.operation.name."""

    def test_realtime_llm_span_operation_name(self):
        """Realtime LLM span must set gen_ai.operation.name."""
        from opentelemetry.instrumentation.openai_agents._realtime_wrappers import (
            RealtimeTracingState,
        )

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        state = RealtimeTracingState(tracer)
        state.start_workflow_span("TestAgent")
        state.start_agent_span("TestAgent")

        state.record_prompt("user", "Hello")

        with patch(
            "opentelemetry.instrumentation.openai_agents._realtime_wrappers.should_send_prompts",
            return_value=True,
        ):
            state.create_llm_span("Hi there!")

        state.cleanup()
        state.end_workflow_span()

        spans = exporter.get_finished_spans()
        llm_span = next((s for s in spans if s.name == "openai.realtime"), None)
        assert llm_span is not None

        op_name = llm_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        assert op_name is not None, "gen_ai.operation.name must be set on realtime LLM span"
        # "realtime" is a custom extension (no well-known OTel equivalent);
        # lock the current value so changes are intentional.
        assert op_name == "realtime", (
            f"Expected 'realtime' operation name, got '{op_name}'"
        )

    def test_realtime_audio_span_operation_name(self):
        """Realtime audio span must set gen_ai.operation.name."""
        from opentelemetry.instrumentation.openai_agents._realtime_wrappers import (
            RealtimeTracingState,
        )

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        state = RealtimeTracingState(tracer)
        state.start_workflow_span("TestAgent")
        state.start_agent_span("TestAgent")

        state.start_audio_span("item-1", 0)
        state.end_audio_span("item-1", 0)

        state.cleanup()
        state.end_workflow_span()

        spans = exporter.get_finished_spans()
        audio_span = next(
            (s for s in spans if s.name == "openai.realtime" and
             s.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "realtime"),
            None,
        )
        assert audio_span is not None, "Audio span must exist with operation name 'realtime'"


# ---------------------------------------------------------------------------
# Realtime provider name
# ---------------------------------------------------------------------------

class TestRealtimeProviderName:
    """Verify realtime spans use gen_ai.provider.name."""

    def test_realtime_workflow_uses_provider_name(self):
        from opentelemetry.instrumentation.openai_agents._realtime_wrappers import (
            RealtimeTracingState,
        )

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        state = RealtimeTracingState(tracer)
        state.start_workflow_span("TestAgent")
        state.end_workflow_span()

        spans = exporter.get_finished_spans()
        wf_span = next((s for s in spans if s.name == "Realtime Session"), None)
        assert wf_span is not None

        attrs = dict(wf_span.attributes)
        assert GenAIAttributes.GEN_AI_PROVIDER_NAME in attrs
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"

    def test_realtime_agent_span_uses_provider_name(self):
        from opentelemetry.instrumentation.openai_agents._realtime_wrappers import (
            RealtimeTracingState,
        )

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        state = RealtimeTracingState(tracer)
        state.start_workflow_span("TestAgent")
        state.start_agent_span("TestAgent")
        state.cleanup()
        state.end_workflow_span()

        spans = exporter.get_finished_spans()
        agent_span = next((s for s in spans if s.name == "TestAgent.agent"), None)
        assert agent_span is not None

        attrs = dict(agent_span.attributes)
        assert GenAIAttributes.GEN_AI_PROVIDER_NAME in attrs
        assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"


# ---------------------------------------------------------------------------
# No deprecated gen_ai.system anywhere
# ---------------------------------------------------------------------------

class TestNoDeprecatedAttributes:
    """Ensure no span uses the deprecated gen_ai.system attribute."""

    def test_no_gen_ai_system_in_generation_span(self, tracer_and_exporter):
        """Spans must not contain the deprecated gen_ai.system attribute."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )
        from agents import GenerationSpanData

        tracer, exporter = tracer_and_exporter
        proc = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-dep-1"
        proc.on_trace_start(mock_trace)

        gen_data = GenerationSpanData(model="gpt-4o", model_config={})
        span = MockAgentSpan(gen_data, trace_id="test-dep-1")

        proc.on_span_start(span)
        proc.on_span_end(span)
        proc.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        for s in spans:
            attrs = dict(s.attributes)
            assert "gen_ai.system" not in attrs, (
                f"Span '{s.name}' uses deprecated 'gen_ai.system' attribute. "
                f"Must use 'gen_ai.provider.name' instead."
            )


# ---------------------------------------------------------------------------
# P3: Content gating – trace_content=False must suppress content attributes
# ---------------------------------------------------------------------------

class TestContentGating:
    """Verify opt-in content attributes are not emitted when tracing is disabled."""

    def test_input_messages_suppressed_when_tracing_disabled(self, tracer_and_exporter):
        """gen_ai.input.messages must NOT be set when trace_content=False."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{"role": "user", "content": "secret prompt"}]
        _extract_prompt_attributes(span, input_data, trace_content=False)

        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
        span.end()

    def test_output_messages_suppressed_when_tracing_disabled(self, tracer_and_exporter):
        """gen_ai.output.messages must NOT be set when trace_content=False."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        response = MagicMock()
        response.temperature = None
        response.max_output_tokens = None
        response.top_p = None
        response.model = "gpt-4o"
        response.id = "resp_1"
        response.frequency_penalty = None
        response.finish_reason = "stop"

        content_item = MagicMock()
        content_item.type = "output_text"
        content_item.text = "secret output"

        output_msg = MagicMock()
        output_msg.type = "message"
        output_msg.content = [content_item]
        output_msg.role = "assistant"
        output_msg.name = None  # Not a tool call

        response.output = [output_msg]
        response.usage = None
        response.tools = None

        _extract_response_attributes(span, response, trace_content=False)

        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
        # finish_reasons should still be set (not content-gated)
        assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS in span.attributes
        span.end()

    def test_tool_definitions_suppressed_when_tracing_disabled(
        self, tracer_and_exporter
    ):
        """gen_ai.tool.definitions must NOT be set when trace_content=False."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )
        from agents import GenerationSpanData

        tracer, exporter = tracer_and_exporter
        proc = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-gate-tools"
        proc.on_trace_start(mock_trace)

        gen_data = GenerationSpanData(model="gpt-4o", model_config={})
        span_obj = MockAgentSpan(gen_data, trace_id="test-gate-tools")

        func_mock = MagicMock()
        func_mock.name = "lookup"
        func_mock.description = "Look something up"
        func_mock.parameters = {"type": "object"}

        tool_mock = MagicMock()
        tool_mock.function = func_mock
        tool_mock.type = "function"

        response_mock = MagicMock()
        response_mock.tools = [tool_mock]
        response_mock.output = []
        response_mock.usage = None
        response_mock.temperature = None
        response_mock.max_output_tokens = None
        response_mock.top_p = None
        response_mock.model = "gpt-4o"
        response_mock.id = "resp_1"
        response_mock.frequency_penalty = None
        response_mock.finish_reason = None
        gen_data.response = response_mock

        with patch(
            "opentelemetry.instrumentation.openai_agents._hooks.should_send_prompts",
            return_value=False,
        ):
            proc.on_span_start(span_obj)
            proc.on_span_end(span_obj)

        proc.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        response_spans = [s for s in spans if "response" in s.name or "chat" in s.name]
        for s in response_spans:
            assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS not in s.attributes, (
                f"Span '{s.name}' should not have tool definitions when tracing disabled"
            )


# ---------------------------------------------------------------------------
# P3: Invalid tool arguments fallback – must always be object or null
# ---------------------------------------------------------------------------

class TestInvalidToolArgumentsFallback:
    """Ensure _parse_arguments never returns a raw string."""

    def test_invalid_json_returns_wrapped_object(self):
        """Invalid JSON string must produce {_raw: ...} object."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _parse_arguments,
        )

        result = _parse_arguments("not valid json {{{")
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "_raw" in result
        assert result["_raw"] == "not valid json {{{"

    def test_json_array_returns_wrapped_object(self):
        """JSON array string must produce {_raw: ...} object (not a list)."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _parse_arguments,
        )

        result = _parse_arguments('[1, 2, 3]')
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "_raw" in result

    def test_empty_string_returns_none(self):
        """Empty/whitespace string must return None."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _parse_arguments,
        )

        assert _parse_arguments("") is None
        assert _parse_arguments("   ") is None

    def test_none_returns_none(self):
        """None input must return None."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _parse_arguments,
        )

        assert _parse_arguments(None) is None

    def test_numeric_arg_returns_wrapped_object(self):
        """Non-string non-dict input must produce {_raw: ...} object."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _parse_arguments,
        )

        result = _parse_arguments(42)
        assert isinstance(result, dict)
        assert "_raw" in result

    def test_valid_json_dict_returns_dict(self):
        """Valid JSON dict string must parse normally."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _parse_arguments,
        )

        result = _parse_arguments('{"city": "NYC"}')
        assert isinstance(result, dict)
        assert result == {"city": "NYC"}


# ---------------------------------------------------------------------------
# Spec §1: Multimodal content mapping — lock OTel part types
# Ref: openllmetry-semconv-review.md §1 "Provider-Specific Content Block Mapping"
#   OpenAI image_url  → OTel UriPart  {type: "uri", modality: "image", uri: "..."}
#   OpenAI input_audio → OTel BlobPart {type: "blob", modality: "audio", ...}
# ---------------------------------------------------------------------------

class TestMultimodalInputMapping:
    """Lock multimodal content blocks to OTel part types per spec."""

    def test_image_url_maps_to_uri_part(self, tracer_and_exporter):
        """Spec §1: OpenAI image_url MUST map to UriPart, NOT 'image_url' type."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.png"},
                },
            ],
        }]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)
        parts = messages[0]["parts"]

        assert len(parts) == 2
        assert parts[0] == {"type": "text", "content": "What is in this image?"}
        # Spec: UriPart — NOT {"type": "image_url", ...}
        assert parts[1]["type"] == "uri", (
            f"image_url must map to UriPart (type='uri'), got type='{parts[1]['type']}'"
        )
        assert parts[1]["modality"] == "image"
        assert parts[1]["uri"] == "https://example.com/img.png"

        span.end()

    def test_input_audio_maps_to_blob_part(self, tracer_and_exporter):
        """Spec §1: OpenAI input_audio MUST map to BlobPart, NOT 'input_audio' type."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": "base64audiodata==", "format": "wav"},
                },
            ],
        }]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)
        parts = messages[0]["parts"]

        assert len(parts) == 1
        # Spec: BlobPart — NOT {"type": "input_audio", ...}
        assert parts[0]["type"] == "blob", (
            f"input_audio must map to BlobPart (type='blob'), got type='{parts[0]['type']}'"
        )
        assert parts[0]["modality"] == "audio"
        assert parts[0]["content"] == "base64audiodata=="

        span.end()

    def test_mixed_text_blocks_mapped(self, tracer_and_exporter):
        """Spec §1: Multiple text blocks → multiple TextPart objects."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "First paragraph."},
                {"type": "text", "text": "Second paragraph."},
            ],
        }]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)
        parts = messages[0]["parts"]

        assert len(parts) == 2
        assert parts[0] == {"type": "text", "content": "First paragraph."}
        assert parts[1] == {"type": "text", "content": "Second paragraph."}

        span.end()

    def test_plain_string_content_produces_text_part(self, tracer_and_exporter):
        """Spec §1: Plain string content → single TextPart."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{"role": "user", "content": "Hello"}]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)
        parts = messages[0]["parts"]

        assert len(parts) == 1
        assert parts[0] == {"type": "text", "content": "Hello"}

        span.end()

    def test_text_key_is_content_not_text(self, tracer_and_exporter):
        """Spec §1: TextPart key is 'content', NOT 'text'."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{"role": "user", "content": "Check key name"}]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)
        part = messages[0]["parts"][0]

        assert "content" in part, "TextPart must use 'content' key"
        assert "text" not in part, (
            "TextPart must NOT use 'text' key — spec requires 'content'"
        )

        span.end()

    def test_unknown_block_type_preserved_as_generic_part(self, tracer_and_exporter):
        """Spec §1: Unknown block types → GenericPart with type preserved."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{
            "role": "user",
            "content": [
                {"type": "custom_widget", "widget_id": "w1"},
            ],
        }]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)
        part = messages[0]["parts"][0]

        assert part["type"] == "custom_widget", "Unknown type must be preserved"

        span.end()

    def test_sdk_object_image_url_maps_to_uri_part(self, tracer_and_exporter):
        """Spec §1: SDK-object image_url blocks also map to UriPart."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _content_block_to_part,
        )

        url_obj = MagicMock()
        url_obj.url = "https://example.com/photo.jpg"
        block = MagicMock()
        block.type = "image_url"
        block.image_url = url_obj

        result = _content_block_to_part(block)

        assert result["type"] == "uri"
        assert result["modality"] == "image"
        assert result["uri"] == "https://example.com/photo.jpg"

    def test_sdk_object_input_audio_maps_to_blob_part(self, tracer_and_exporter):
        """Spec §1: SDK-object input_audio blocks also map to BlobPart."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _content_block_to_part,
        )

        audio_obj = MagicMock()
        audio_obj.data = "base64data=="
        block = MagicMock()
        block.type = "input_audio"
        block.input_audio = audio_obj

        result = _content_block_to_part(block)

        assert result["type"] == "blob"
        assert result["modality"] == "audio"
        assert result["content"] == "base64data=="

    def test_unknown_block_preserves_per_field_structure(self):
        """Unknown block types must preserve per-field structure, not json.dumps the whole block."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _dict_block_to_part,
        )

        block = {"type": "file", "file_id": "file_abc123", "filename": "data.csv"}
        part = _dict_block_to_part(block)

        assert part["type"] == "file"
        assert "file_id" in part, f"Expected 'file_id' in part, got: {part}"
        assert part["file_id"] == "file_abc123"


# ---------------------------------------------------------------------------
# Spec §1: Assistant text + tool_calls combined
# Ref: "Messages can include both text and tool_call parts"
# ---------------------------------------------------------------------------

class TestAssistantTextWithToolCalls:
    """Lock: assistant messages with both text and tool_calls emit both parts."""

    def test_text_and_tool_call_both_present(self, tracer_and_exporter):
        """Spec §1: text content alongside tool_calls → text + tool_call parts."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{
            "role": "assistant",
            "content": "Let me look that up.",
            "tool_calls": [{
                "id": "call_1",
                "function": {
                    "name": "search",
                    "arguments": '{"q": "weather"}',
                },
            }],
        }]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)
        parts = messages[0]["parts"]

        types = [p["type"] for p in parts]
        assert "text" in types, "Missing text part alongside tool_call"
        assert "tool_call" in types, "Missing tool_call part"

        text_part = next(p for p in parts if p["type"] == "text")
        assert text_part["content"] == "Let me look that up."

        tc_part = next(p for p in parts if p["type"] == "tool_call")
        assert tc_part["name"] == "search"
        assert isinstance(tc_part["arguments"], dict)

        span.end()

    def test_tool_calls_without_content(self, tracer_and_exporter):
        """Spec §1: tool_calls with no text content → only tool_call parts."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "search", "arguments": "{}"},
            }],
        }]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)
        parts = messages[0]["parts"]

        assert len(parts) == 1
        assert parts[0]["type"] == "tool_call"

        span.end()


# ---------------------------------------------------------------------------
# Spec §1/§4: Output messages — non-text parts, finish_reason always present
# Ref: "finish_reason in output JSON: required per schema — always set"
# ---------------------------------------------------------------------------

class TestOutputNonTextParts:
    """Lock: output messages handle refusal, reasoning, and finish_reason."""

    def test_refusal_content_mapped(self, tracer_and_exporter):
        """Spec §1: Refusal content → {type: 'text', content: '...'} (standard TextPart)."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        response = MagicMock()
        response.temperature = None
        response.max_output_tokens = None
        response.top_p = None
        response.model = "gpt-4o"
        response.id = "resp_1"
        response.frequency_penalty = None
        response.finish_reason = "stop"

        content_item = MagicMock()
        content_item.type = "refusal"
        content_item.refusal = "I cannot help with that."
        content_item.text = None

        output_msg = MagicMock()
        output_msg.type = "message"
        output_msg.content = [content_item]
        output_msg.role = "assistant"
        output_msg.name = None

        response.output = [output_msg]
        response.usage = None
        response.tools = None

        _extract_response_attributes(span, response, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        messages = json.loads(raw)
        parts = messages[0]["parts"]

        assert len(parts) == 1
        assert parts[0]["type"] == "refusal"
        assert parts[0]["content"] == "I cannot help with that."

        span.end()

    def test_output_finish_reason_always_present_in_json(self, tracer_and_exporter):
        """Spec §4: finish_reason key MUST always exist in output JSON (even if unknown)."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        response = MagicMock()
        response.temperature = None
        response.max_output_tokens = None
        response.top_p = None
        response.model = "gpt-4o"
        response.id = "resp_1"
        response.frequency_penalty = None
        response.finish_reason = None  # Unknown/absent
        response.status = None

        content_item = MagicMock()
        content_item.type = "output_text"
        content_item.text = "Hello"

        output_msg = MagicMock()
        output_msg.type = "message"
        output_msg.content = [content_item]
        output_msg.role = "assistant"
        output_msg.name = None

        response.output = [output_msg]
        response.usage = None

        _extract_response_attributes(span, response, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        messages = json.loads(raw)

        assert "finish_reason" in messages[0], (
            "finish_reason key must always be present in output JSON per schema"
        )
        assert messages[0]["finish_reason"] == ""

        span.end()

    def test_output_finish_reason_mapped_value(self, tracer_and_exporter):
        """Spec §4: finish_reason in JSON uses mapped OTel value."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        response = MagicMock()
        response.temperature = None
        response.max_output_tokens = None
        response.top_p = None
        response.model = "gpt-4o"
        response.id = "resp_1"
        response.frequency_penalty = None
        response.finish_reason = "tool_calls"  # OpenAI plural

        content_item = MagicMock()
        content_item.type = "output_text"
        content_item.text = "Calling tool"

        output_msg = MagicMock()
        output_msg.type = "message"
        output_msg.content = [content_item]
        output_msg.role = "assistant"
        output_msg.name = None

        response.output = [output_msg]
        response.usage = None

        _extract_response_attributes(span, response, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        messages = json.loads(raw)

        # Spec §4: tool_calls → tool_call (singular)
        assert messages[0]["finish_reason"] == "tool_call"

        span.end()

    def test_reasoning_content_mapped(self, tracer_and_exporter):
        """Spec §1: Reasoning content → {type: 'reasoning', content: '...'}."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        response = MagicMock()
        response.temperature = None
        response.max_output_tokens = None
        response.top_p = None
        response.model = "gpt-4o"
        response.id = "resp_1"
        response.frequency_penalty = None
        response.finish_reason = "stop"

        content_item = MagicMock()
        content_item.type = "reasoning"
        content_item.text = None

        summary_item = MagicMock()
        summary_item.text = "The user wants weather info"
        content_item.summary = [summary_item]

        output_msg = MagicMock()
        output_msg.type = "message"
        output_msg.content = [content_item]
        output_msg.role = "assistant"
        output_msg.name = None

        response.output = [output_msg]
        response.usage = None

        _extract_response_attributes(span, response, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        messages = json.loads(raw)
        parts = messages[0]["parts"]

        assert len(parts) == 1
        assert parts[0]["type"] == "reasoning"
        assert "weather" in parts[0]["content"]

        span.end()

    def test_reasoning_summary_dict_items_extract_text(self, tracer_and_exporter):
        """Dict-form reasoning summary items must extract 'text' field, not dump repr."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )
        from types import SimpleNamespace

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        content_item = SimpleNamespace(
            type="reasoning",
            summary=[{"text": "The model considered options."}],
        )
        output = SimpleNamespace(
            type="message", content=[content_item], role="assistant",
        )
        response = SimpleNamespace(
            temperature=None, max_output_tokens=None, top_p=None,
            model=None, id=None, frequency_penalty=None,
            finish_reason=None, status="completed",
            output=[output], usage=None,
        )

        _extract_response_attributes(span, response, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        messages = json.loads(raw)
        reasoning_parts = [
            p for msg in messages for p in msg.get("parts", [])
            if p.get("type") == "reasoning"
        ]
        assert len(reasoning_parts) >= 1
        assert "{'text'" not in reasoning_parts[0]["content"], (
            "Dict repr leaked into reasoning content"
        )
        assert "The model considered options" in reasoning_parts[0]["content"]

        span.end()


# ---------------------------------------------------------------------------
# Spec §2: Roles — only OTel-valid roles emitted
# ---------------------------------------------------------------------------

class TestRoles:
    """Lock: only valid OTel roles (system, user, assistant, tool) emitted."""

    def test_system_role_preserved(self, tracer_and_exporter):
        """Spec §2: system role kept inline in input messages for OpenAI."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)

        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

        span.end()

    def test_developer_role_preserved(self, tracer_and_exporter):
        """Spec §2: provider-specific roles like 'developer' are allowed."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{"role": "developer", "content": "Be concise."}]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)

        assert messages[0]["role"] == "developer"

        span.end()


# ---------------------------------------------------------------------------
# Spec §4: finish_reasons top-level span attribute — comprehensive
# ---------------------------------------------------------------------------

class TestFinishReasonTopLevel:
    """Lock: gen_ai.response.finish_reasons as top-level span attribute."""

    def test_finish_reasons_not_gated_by_content(self, tracer_and_exporter):
        """Spec §4: finish_reasons set even when should_send_prompts()=False."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        response = MagicMock()
        response.temperature = None
        response.max_output_tokens = None
        response.top_p = None
        response.model = "gpt-4o"
        response.id = "resp_1"
        response.frequency_penalty = None
        response.finish_reason = "stop"
        response.output = []
        response.usage = None

        _extract_response_attributes(span, response, trace_content=False)

        # finish_reasons is metadata, NOT content — must be set
        assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS in span.attributes
        assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
            "stop",
        )
        # But output messages must NOT be set
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes

        span.end()

    def test_none_finish_reason_omits_attribute(self, tracer_and_exporter):
        """Spec §4: None finish_reason → attribute omitted, NOT fabricated."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        response = MagicMock()
        response.temperature = None
        response.max_output_tokens = None
        response.top_p = None
        response.model = "gpt-4o"
        response.id = "resp_1"
        response.frequency_penalty = None
        response.finish_reason = None
        response.output = []
        response.usage = None

        _extract_response_attributes(span, response, trace_content=True)

        assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in span.attributes

        span.end()


# ---------------------------------------------------------------------------
# Spec §1: _msg_to_dict with SDK objects (not just dicts)
# ---------------------------------------------------------------------------

class TestMsgToDict:
    """Lock: _msg_to_dict normalizes SDK objects to plain dicts."""

    def test_sdk_object_normalized(self):
        """Spec §1: SDK objects with attributes are normalized to dicts."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _msg_to_dict,
        )

        obj = MagicMock()
        obj.role = "user"
        obj.content = "Hello"
        # Only set some attrs
        del obj.tool_call_id
        del obj.tool_calls

        result = _msg_to_dict(obj)
        assert isinstance(result, dict)
        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_dict_passed_through(self):
        """Spec §1: dict messages are returned as-is."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _msg_to_dict,
        )

        msg = {"role": "user", "content": "Hello"}
        result = _msg_to_dict(msg)
        assert result is msg  # Same reference, not a copy


# ---------------------------------------------------------------------------
# Spec §1: Tool call round-trip (request → response)
# ---------------------------------------------------------------------------

class TestToolCallRoundTrip:
    """Lock: tool_call → tool_call_response forms a complete round trip."""

    def test_full_round_trip(self, tracer_and_exporter):
        """Spec §1: tool_call request and response correlate via id."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_abc",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "NYC"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": '{"temp": 72}',
            },
        ]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)

        # Message 1: assistant with tool_call
        assert messages[0]["role"] == "assistant"
        tc = messages[0]["parts"][0]
        assert tc["type"] == "tool_call"
        assert tc["id"] == "call_abc"
        assert tc["name"] == "get_weather"
        assert tc["arguments"] == {"city": "NYC"}

        # Message 2: tool response correlating via same id
        assert messages[1]["role"] == "tool"
        resp = messages[1]["parts"][0]
        assert resp["type"] == "tool_call_response"
        assert resp["id"] == "call_abc"
        assert resp["response"] == '{"temp": 72}'

        span.end()


# ---------------------------------------------------------------------------
# Spec: _convert_agents_sdk_message unknown type returns (None, [])
# ---------------------------------------------------------------------------

class TestAgentsSdkUnknownType:
    """Lock: unknown Agents SDK message types are silently skipped."""

    def test_unknown_type_skipped(self, tracer_and_exporter):
        """Unknown Agents SDK type must not produce a message."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{"type": "unknown_sdk_type", "data": "foo"}]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        # No messages should be set (unknown type skipped)
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes

        span.end()


# ---------------------------------------------------------------------------
# P1-1: input_text / output_text blocks must map to TextPart in input path
# ---------------------------------------------------------------------------

class TestInputTextOutputTextMapping:
    """Verify Responses API input_text/output_text blocks map to TextPart."""

    def test_dict_input_text_maps_to_text_part(self):
        """input_text dict block must produce {type: 'text', content: '...'}."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _dict_block_to_part,
        )

        block = {"type": "input_text", "text": "Hello from user"}
        result = _dict_block_to_part(block)

        assert result["type"] == "text", (
            f"input_text should map to type='text', got '{result['type']}'"
        )
        assert result["content"] == "Hello from user", (
            f"input_text content should be the text value, got '{result.get('content')}'"
        )
        assert "data" not in result, (
            "input_text should NOT fall through to generic path with 'data' key"
        )

    def test_dict_output_text_maps_to_text_part(self):
        """output_text dict block must produce {type: 'text', content: '...'}."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _dict_block_to_part,
        )

        block = {"type": "output_text", "text": "Here is my response"}
        result = _dict_block_to_part(block)

        assert result["type"] == "text", (
            f"output_text should map to type='text', got '{result['type']}'"
        )
        assert result["content"] == "Here is my response"
        assert "data" not in result

    def test_object_input_text_maps_to_text_part(self):
        """input_text SDK object must produce {type: 'text', content: '...'}."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _object_block_to_part,
        )

        block = MagicMock()
        block.type = "input_text"
        block.text = "Hello from user"

        result = _object_block_to_part(block)

        assert result["type"] == "text", (
            f"input_text object should map to type='text', got '{result['type']}'"
        )
        assert result["content"] == "Hello from user"
        assert "data" not in result

    def test_object_output_text_maps_to_text_part(self):
        """output_text SDK object must produce {type: 'text', content: '...'}."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _object_block_to_part,
        )

        block = MagicMock()
        block.type = "output_text"
        block.text = "Here is my response"

        result = _object_block_to_part(block)

        assert result["type"] == "text", (
            f"output_text object should map to type='text', got '{result['type']}'"
        )
        assert result["content"] == "Here is my response"
        assert "data" not in result

    def test_input_text_in_full_input_message_pipeline(self, tracer_and_exporter):
        """input_text blocks in chat messages must produce valid TextPart in gen_ai.input.messages."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [
            {"role": "user", "content": [{"type": "input_text", "text": "Hello, can you help me?"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Of course!"}]},
        ]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert raw is not None
        messages = json.loads(raw)

        # User message: input_text → TextPart
        user_parts = messages[0]["parts"]
        assert user_parts[0]["type"] == "text", (
            f"input_text in pipeline should be type='text', got '{user_parts[0]['type']}'"
        )
        assert user_parts[0]["content"] == "Hello, can you help me?"

        # Assistant message: output_text → TextPart
        assistant_parts = messages[1]["parts"]
        assert assistant_parts[0]["type"] == "text", (
            f"output_text in pipeline should be type='text', got '{assistant_parts[0]['type']}'"
        )
        assert assistant_parts[0]["content"] == "Of course!"

        span.end()


# ---------------------------------------------------------------------------
# P2 (was P1-2): gen_ai.request.model set at span start from span_data.model
# ---------------------------------------------------------------------------

class TestRequestModelAtSpanStart:
    """Verify gen_ai.request.model is set at span creation from span_data."""

    def test_request_model_set_from_span_data(self, tracer_and_exporter):
        """gen_ai.request.model must be set at span start from span_data.model."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )
        from agents import GenerationSpanData

        tracer, exporter = tracer_and_exporter
        proc = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-reqmodel-1"
        proc.on_trace_start(mock_trace)

        gen_data = GenerationSpanData(model="gpt-4o-mini", model_config={})
        span = MockAgentSpan(gen_data, trace_id="test-reqmodel-1")

        proc.on_span_start(span)
        # Don't call on_span_end — check span attributes right after creation
        otel_span = proc._otel_spans.get(span)
        assert otel_span is not None, "OTel span should exist after on_span_start"

        attrs = dict(otel_span.attributes)
        assert GenAIAttributes.GEN_AI_REQUEST_MODEL in attrs, (
            f"gen_ai.request.model should be set at span start, got keys: {list(attrs.keys())}"
        )
        assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"

        # Clean up
        proc.on_span_end(span)
        proc.on_trace_end(mock_trace)

    def test_request_model_fallback_when_response_model_missing(self, tracer_and_exporter):
        """gen_ai.request.model must persist even if response.model is None."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )
        from agents import GenerationSpanData

        tracer, exporter = tracer_and_exporter
        proc = OpenTelemetryTracingProcessor(tracer)

        mock_trace = MagicMock()
        mock_trace.trace_id = "test-reqmodel-2"
        proc.on_trace_start(mock_trace)

        gen_data = GenerationSpanData(model="gpt-4o", model_config={})
        # Simulate a response with no model
        gen_data.response = MagicMock()
        gen_data.response.model = None
        gen_data.response.id = None
        gen_data.response.temperature = None
        gen_data.response.max_output_tokens = None
        gen_data.response.top_p = None
        gen_data.response.frequency_penalty = None
        gen_data.response.finish_reason = None
        gen_data.response.output = []
        gen_data.response.usage = None
        gen_data.response.tools = []

        span = MockAgentSpan(gen_data, trace_id="test-reqmodel-2")
        proc.on_span_start(span)
        proc.on_span_end(span)
        proc.on_trace_end(mock_trace)

        spans = exporter.get_finished_spans()
        response_span = next((s for s in spans if s.name == "openai.response"), None)
        assert response_span is not None

        attrs = dict(response_span.attributes)
        assert GenAIAttributes.GEN_AI_REQUEST_MODEL in attrs, (
            "gen_ai.request.model should be set even when response.model is None"
        )
        assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o"


# ---------------------------------------------------------------------------
# P2-1: Tool-call response parts include response key even when content=None
# ---------------------------------------------------------------------------

class TestToolResponseNoneContent:
    """Verify tool_call_response includes response key when content is None."""

    def test_tool_response_part_has_response_key_when_none(self):
        """tool_call_response must include 'response' key even when content is None."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _build_tool_response_part,
        )

        part = _build_tool_response_part("call_123", None)

        assert part["type"] == "tool_call_response"
        assert part["id"] == "call_123"
        assert "response" in part, (
            "tool_call_response must include 'response' key even when content is None"
        )
        assert part["response"] == ""

    def test_tool_response_part_has_response_key_when_present(self):
        """tool_call_response with content must include 'response' key."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _build_tool_response_part,
        )

        part = _build_tool_response_part("call_456", "72 degrees")

        assert part["type"] == "tool_call_response"
        assert part["id"] == "call_456"
        assert part["response"] == "72 degrees"

    def test_tool_response_none_content_in_full_pipeline(self, tracer_and_exporter):
        """Tool message with content=None must still produce response key in gen_ai.input.messages."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [
            {"role": "tool", "tool_call_id": "call_789", "content": None},
        ]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert raw is not None
        messages = json.loads(raw)

        tool_part = messages[0]["parts"][0]
        assert tool_part["type"] == "tool_call_response"
        assert "response" in tool_part, (
            "tool_call_response must include 'response' key even with None content"
        )

        span.end()

    def test_structured_dict_result_preserved(self):
        """Dict tool result should be kept as-is, not stringified."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _build_tool_response_part,
        )

        part = _build_tool_response_part("call_1", {"status": "ok", "count": 5})
        assert isinstance(part["response"], dict)
        assert part["response"] == {"status": "ok", "count": 5}

    def test_structured_list_result_preserved(self):
        """List tool result should be kept as-is, not stringified."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _build_tool_response_part,
        )

        part = _build_tool_response_part("call_2", [1, 2, 3])
        assert isinstance(part["response"], list)


# ---------------------------------------------------------------------------
# P2-2: Realtime LLM spans set response metadata
# ---------------------------------------------------------------------------

class TestRealtimeResponseMetadata:
    """Verify realtime LLM spans set recommended response attributes."""

    def test_realtime_span_sets_response_model(self, tracer_and_exporter):
        """Realtime LLM spans should set gen_ai.response.model."""
        from opentelemetry.instrumentation.openai_agents._realtime_wrappers import (
            RealtimeTracingState,
        )

        tracer, exporter = tracer_and_exporter
        state = RealtimeTracingState(tracer)
        state.model_name = "gpt-4o-realtime-preview-2024-12-17"

        # Create a parent span for context
        parent = tracer.start_span("parent")
        state.pending_prompts.append(("user", "Hello"))
        state.prompt_start_time = 1000

        state.create_llm_span("Hi there!")
        parent.end()

        finished = exporter.get_finished_spans()
        rt_span = next((s for s in finished if s.name == "openai.realtime"), None)
        assert rt_span is not None

        attrs = dict(rt_span.attributes)
        assert GenAIAttributes.GEN_AI_RESPONSE_MODEL in attrs, (
            "Realtime LLM span should set gen_ai.response.model"
        )

    def test_realtime_span_sets_finish_reason_empty(self, tracer_and_exporter):
        """Realtime LLM spans should use '' finish_reason, NOT fabricate 'stop'."""
        from opentelemetry.instrumentation.openai_agents._realtime_wrappers import (
            RealtimeTracingState,
        )

        tracer, exporter = tracer_and_exporter
        state = RealtimeTracingState(tracer)
        state.model_name = "gpt-4o-realtime-preview"

        parent = tracer.start_span("parent")
        state.pending_prompts.append(("user", "Hello"))
        state.prompt_start_time = 1000

        with patch(
            "opentelemetry.instrumentation.openai_agents._realtime_wrappers.should_send_prompts",
            return_value=True,
        ):
            state.create_llm_span("Hi there!")

        parent.end()

        finished = exporter.get_finished_spans()
        rt_span = next((s for s in finished if s.name == "openai.realtime"), None)
        assert rt_span is not None

        raw = rt_span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        assert raw is not None
        messages = json.loads(raw)
        assert messages[0]["finish_reason"] == "", (
            f"Realtime finish_reason should be '' (not fabricated), got '{messages[0].get('finish_reason')}'"
        )


# ---------------------------------------------------------------------------
# F1: BlobPart must use "content" key, NOT "data"
# OTel spec: BlobPart.required = ["type", "modality", "content"]
# Upstream refs: opentelemetry-python-contrib Blob dataclass uses "content",
#   Bedrock/OpenAI instrumentations use "content" for blob parts.
# ---------------------------------------------------------------------------

class TestBlobPartContentKey:
    """F1: BlobPart must use 'content' key per OTel GenAI semconv."""

    def test_dict_input_audio_blob_uses_content_key(self, tracer_and_exporter):
        """Dict input_audio block must produce BlobPart with 'content', NOT 'data'."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _dict_block_to_part,
        )

        block = {
            "type": "input_audio",
            "input_audio": {"data": "base64audiodata==", "format": "wav"},
        }
        result = _dict_block_to_part(block)

        assert result["type"] == "blob"
        assert result["modality"] == "audio"
        assert "content" in result, (
            "BlobPart must use 'content' key per OTel spec, not 'data'"
        )
        assert "data" not in result, (
            "BlobPart must NOT use 'data' key — spec requires 'content'"
        )
        assert result["content"] == "base64audiodata=="

    def test_object_input_audio_blob_uses_content_key(self, tracer_and_exporter):
        """SDK-object input_audio block must produce BlobPart with 'content', NOT 'data'."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _object_block_to_part,
        )

        audio_obj = MagicMock()
        audio_obj.data = "base64data=="
        block = MagicMock()
        block.type = "input_audio"
        block.input_audio = audio_obj

        result = _object_block_to_part(block)

        assert result["type"] == "blob"
        assert result["modality"] == "audio"
        assert "content" in result, (
            "BlobPart must use 'content' key per OTel spec, not 'data'"
        )
        assert "data" not in result, (
            "BlobPart must NOT use 'data' key — spec requires 'content'"
        )
        assert result["content"] == "base64data=="

    def test_blob_content_key_in_full_pipeline(self, tracer_and_exporter):
        """BlobPart 'content' key must survive through the full input message pipeline."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_prompt_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        input_data = [{
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": "YXVkaW9kYXRh", "format": "mp3"},
                },
            ],
        }]
        _extract_prompt_attributes(span, input_data, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        messages = json.loads(raw)
        blob_part = messages[0]["parts"][0]

        assert blob_part["type"] == "blob"
        assert "content" in blob_part, "BlobPart must use 'content' in pipeline output"
        assert "data" not in blob_part, "BlobPart must NOT use 'data' in pipeline output"
        assert blob_part["content"] == "YXVkaW9kYXRh"

        span.end()


# ---------------------------------------------------------------------------
# F2: gen_ai.tool.call.arguments/result must use json.dumps(), NOT str()
# str(dict) produces Python repr with single quotes — not valid JSON.
# All other structured attributes in this package use json.dumps().
# ---------------------------------------------------------------------------

class TestToolCallArgumentsSerialization:
    """F2: Tool call arguments/result must be valid JSON, not Python repr."""

    def test_dict_input_serialized_as_json(self, tracer_and_exporter):
        """Dict tool input must be serialized with json.dumps(), not str()."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )
        from agents import FunctionSpanData

        tracer, _ = tracer_and_exporter
        proc = OpenTelemetryTracingProcessor(tracer)

        func_data = FunctionSpanData(
            name="get_weather",
            input={"city": "London"},
            output="72F",
        )
        otel_span = proc._start_function_span(func_data, parent_context=None)
        proc._end_function_span(otel_span, func_data, trace_content=True)
        otel_span.end()

        from opentelemetry.semconv._incubating.attributes import (
            gen_ai_attributes as GenAIAttributes,
        )
        raw_args = otel_span.attributes[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS]
        # Must be valid JSON (double quotes), NOT Python repr (single quotes)
        assert '"city"' in raw_args, (
            f"Expected JSON with double quotes, got: {raw_args}"
        )
        assert "'" not in raw_args or raw_args == raw_args, (
            f"str() produces single quotes; expected json.dumps(): {raw_args}"
        )
        # Must parse as valid JSON
        parsed = json.loads(raw_args)
        assert parsed == {"city": "London"}

    def test_dict_output_serialized_as_json(self, tracer_and_exporter):
        """Dict tool output must be serialized with json.dumps(), not str()."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )
        from agents import FunctionSpanData

        tracer, _ = tracer_and_exporter
        proc = OpenTelemetryTracingProcessor(tracer)

        func_data = FunctionSpanData(
            name="get_weather",
            input="query",
            output={"temp": 72, "unit": "F"},
        )
        otel_span = proc._start_function_span(func_data, parent_context=None)
        proc._end_function_span(otel_span, func_data, trace_content=True)
        otel_span.end()

        from opentelemetry.semconv._incubating.attributes import (
            gen_ai_attributes as GenAIAttributes,
        )
        raw_result = otel_span.attributes[GenAIAttributes.GEN_AI_TOOL_CALL_RESULT]
        parsed = json.loads(raw_result)
        assert parsed == {"temp": 72, "unit": "F"}

    def test_string_input_kept_as_is(self, tracer_and_exporter):
        """String tool input must be kept as-is (already a string)."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )
        from agents import FunctionSpanData

        tracer, _ = tracer_and_exporter
        proc = OpenTelemetryTracingProcessor(tracer)

        func_data = FunctionSpanData(
            name="echo",
            input='{"already": "json"}',
            output="done",
        )
        otel_span = proc._start_function_span(func_data, parent_context=None)
        proc._end_function_span(otel_span, func_data, trace_content=True)
        otel_span.end()

        from opentelemetry.semconv._incubating.attributes import (
            gen_ai_attributes as GenAIAttributes,
        )
        raw_args = otel_span.attributes[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS]
        assert raw_args == '{"already": "json"}'

    def test_list_output_serialized_as_json(self, tracer_and_exporter):
        """List tool output must be serialized with json.dumps()."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            OpenTelemetryTracingProcessor,
        )
        from agents import FunctionSpanData

        tracer, _ = tracer_and_exporter
        proc = OpenTelemetryTracingProcessor(tracer)

        func_data = FunctionSpanData(
            name="search",
            input="query",
            output=["result1", "result2"],
        )
        otel_span = proc._start_function_span(func_data, parent_context=None)
        proc._end_function_span(otel_span, func_data, trace_content=True)
        otel_span.end()

        from opentelemetry.semconv._incubating.attributes import (
            gen_ai_attributes as GenAIAttributes,
        )
        raw_result = otel_span.attributes[GenAIAttributes.GEN_AI_TOOL_CALL_RESULT]
        parsed = json.loads(raw_result)
        assert parsed == ["result1", "result2"]


# ---------------------------------------------------------------------------
# F3: Responses API status → finish_reason mapping
# The Responses API uses "status" ("completed"/"failed"/"cancelled"/"incomplete"),
# NOT "finish_reason". Must map status to OTel finish reasons.
# Upstream ref: opentelemetry-python-contrib _finish_reason_from_status()
# ---------------------------------------------------------------------------

class TestResponsesApiStatusMapping:
    """F3: Map Responses API 'status' to finish_reason when finish_reason absent."""

    def test_completed_status_maps_to_stop(self, tracer_and_exporter):
        """Responses API status='completed' must map to finish_reason='stop'."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        response = MockResponse(
            output=[],
            model="gpt-4o",
            usage=MockUsage(),
        )
        # Responses API: no finish_reason, but has status
        del response.finish_reason
        response.status = "completed"

        _extract_response_attributes(span, response, trace_content=True)

        finish_reasons = span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert finish_reasons is not None, (
            "status='completed' must produce gen_ai.response.finish_reasons"
        )
        assert "stop" in finish_reasons, (
            f"status='completed' must map to 'stop', got {finish_reasons}"
        )

        span.end()

    def test_failed_status_maps_to_error(self, tracer_and_exporter):
        """Responses API status='failed' must map to OTel finish_reason='error'."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        response = MockResponse(
            output=[],
            model="gpt-4o",
            usage=MockUsage(),
        )
        del response.finish_reason
        response.status = "failed"

        _extract_response_attributes(span, response, trace_content=True)

        finish_reasons = span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert finish_reasons is not None
        assert "error" in finish_reasons

        span.end()

    def test_cancelled_status_maps_to_error(self, tracer_and_exporter):
        """Responses API status='cancelled' must map to OTel finish_reason='error'."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        response = MockResponse(
            output=[],
            model="gpt-4o",
            usage=MockUsage(),
        )
        del response.finish_reason
        response.status = "cancelled"

        _extract_response_attributes(span, response, trace_content=True)

        finish_reasons = span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert finish_reasons is not None
        assert "error" in finish_reasons

        span.end()

    def test_incomplete_status_maps_to_length(self, tracer_and_exporter):
        """Responses API status='incomplete' must map to OTel finish_reason='length'."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        response = MockResponse(
            output=[],
            model="gpt-4o",
            usage=MockUsage(),
        )
        del response.finish_reason
        response.status = "incomplete"

        _extract_response_attributes(span, response, trace_content=True)

        finish_reasons = span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert finish_reasons is not None
        assert "length" in finish_reasons

        span.end()

    def test_status_not_used_when_finish_reason_present(self, tracer_and_exporter):
        """When finish_reason is present, status must NOT override it."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        response = MockResponse(
            output=[],
            model="gpt-4o",
            usage=MockUsage(),
            finish_reason="stop",
        )
        response.status = "completed"  # Both present — finish_reason wins

        _extract_response_attributes(span, response, trace_content=True)

        finish_reasons = span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert finish_reasons is not None
        assert "stop" in finish_reasons

        span.end()

    def test_completed_status_maps_to_stop_in_output_messages(self, tracer_and_exporter):
        """status='completed' → output message finish_reason='stop'."""
        from opentelemetry.instrumentation.openai_agents._hooks import (
            _extract_response_attributes,
        )

        tracer, _ = tracer_and_exporter
        span = tracer.start_span("test")

        content_item = MockContentItem(text="Done")
        output_item = MockResponseOutput(role="assistant", content=[content_item])
        response = MockResponse(
            output=[output_item],
            model="gpt-4o",
            usage=MockUsage(),
        )
        del response.finish_reason
        response.status = "completed"

        _extract_response_attributes(span, response, trace_content=True)

        raw = span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        messages = json.loads(raw)
        assert messages[0]["finish_reason"] == "stop", (
            f"status='completed' should produce finish_reason='stop' in output, "
            f"got '{messages[0]['finish_reason']}'"
        )

        span.end()


# ---------------------------------------------------------------------------
# Missing finish_reason mapping tests
# ---------------------------------------------------------------------------

class TestFinishReasonMappingCompleteness:
    """Cover finish_reason mappings missing from original test suite."""

    def test_length_mapping(self):
        """'length' must map to 'length'."""
        from opentelemetry.instrumentation.openai_agents._hooks import _map_finish_reason
        assert _map_finish_reason("length") == "length"

    def test_content_filter_mapping(self):
        """'content_filter' must map to 'content_filter'."""
        from opentelemetry.instrumentation.openai_agents._hooks import _map_finish_reason
        assert _map_finish_reason("content_filter") == "content_filter"

    def test_error_mapping(self):
        """'error' must map to 'error'."""
        from opentelemetry.instrumentation.openai_agents._hooks import _map_finish_reason
        assert _map_finish_reason("error") == "error"

    def test_unknown_finish_reason_passes_through(self):
        """Unknown/new finish reason values must pass through unchanged."""
        from opentelemetry.instrumentation.openai_agents._hooks import _map_finish_reason
        assert _map_finish_reason("some_new_reason") == "some_new_reason"

    def test_function_call_maps_to_tool_call(self):
        """Legacy 'function_call' must map to 'tool_call'."""
        from opentelemetry.instrumentation.openai_agents._hooks import _map_finish_reason
        assert _map_finish_reason("function_call") == "tool_call"
