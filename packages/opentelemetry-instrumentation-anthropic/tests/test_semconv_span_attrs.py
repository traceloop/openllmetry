"""
Unit tests for OTel GenAI semantic conventions compliance in span attributes.

These tests verify that span_utils.py emits the new OTel GenAI spec attributes:
  - gen_ai.input.messages  (replaces gen_ai.prompt.{i}.*)
  - gen_ai.output.messages (replaces gen_ai.completion.{i}.*)
  - gen_ai.system_instructions (replaces gen_ai.prompt.0 with role=system)
  - gen_ai.tool.definitions (replaces llm.request.functions.{i}.*)
  - gen_ai.response.finish_reasons (array, replaces per-completion finish_reason)
"""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiOperationNameValues,
    GenAiSystemValues,
)

from opentelemetry.instrumentation.anthropic.span_utils import (
    aset_input_attributes,
    set_response_attributes,
    set_streaming_response_attributes,
)
from opentelemetry.instrumentation.anthropic.utils import TRACELOOP_TRACE_CONTENT


@pytest.fixture(autouse=True)
def enable_content_tracing():
    """Ensure content tracing is enabled for all tests."""
    os.environ[TRACELOOP_TRACE_CONTENT] = "true"
    yield
    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)


def make_span():
    """Create a mock span that collects set_attribute calls."""
    attributes = {}
    span = MagicMock()
    span.attributes = attributes
    span.set_attribute = lambda k, v: attributes.update({k: v})
    span.context.trace_id = 1234567890
    span.context.span_id = 9876543210
    return span


# ---------------------------------------------------------------------------
# Input attribute tests
# ---------------------------------------------------------------------------

def test_input_messages_simple_user_message():
    """gen_ai.input.messages should be a JSON array with role+content."""
    span = make_span()
    kwargs = {
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": "Tell me a joke"}],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in span.attributes, (
        "gen_ai.input.messages must be set"
    )
    messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert messages == [{"role": "user", "content": "Tell me a joke"}]

    # Old attribute must NOT be set
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.content" not in span.attributes
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.role" not in span.attributes


def test_input_messages_multi_turn():
    """gen_ai.input.messages should include all turns."""
    span = make_span()
    kwargs = {
        "model": "claude-3-opus-20240229",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert len(messages) == 3
    assert messages[0] == {"role": "user", "content": "Hello"}
    assert messages[1] == {"role": "assistant", "content": "Hi there!"}
    assert messages[2] == {"role": "user", "content": "How are you?"}


def test_system_instructions_attribute():
    """System prompt should be in gen_ai.system_instructions, NOT as a message."""
    span = make_span()
    kwargs = {
        "model": "claude-3-opus-20240229",
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    # System instructions should be a standalone attribute
    assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS in span.attributes, (
        "gen_ai.system_instructions must be set"
    )
    assert span.attributes[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS] == (
        "You are a helpful assistant."
    )

    # System should NOT appear as part of gen_ai.input.messages
    messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    roles = [m["role"] for m in messages]
    assert "system" not in roles, (
        "System message must not appear inside gen_ai.input.messages"
    )

    # Old attribute must NOT be set
    assert f"{GenAIAttributes.GEN_AI_PROMPT}.0.role" not in span.attributes


def test_tool_definitions_attribute():
    """Tools should be serialised into gen_ai.tool.definitions JSON array."""
    span = make_span()
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]
    kwargs = {
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "tools": tools,
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS in span.attributes, (
        "gen_ai.tool.definitions must be set"
    )
    defs = json.loads(span.attributes[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS])
    assert len(defs) == 1
    assert defs[0]["name"] == "get_weather"
    assert defs[0]["description"] == "Get the current weather"
    assert "input_schema" in defs[0]

    # Old attribute must NOT be set
    from opentelemetry.semconv_ai import SpanAttributes
    assert f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name" not in span.attributes


def test_input_messages_with_tool_calls_in_content():
    """Tool use blocks in assistant messages should be captured as tool_calls."""
    span = make_span()
    kwargs = {
        "model": "claude-3-opus-20240229",
        "messages": [
            {"role": "user", "content": "What's the weather in SF?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "get_weather",
                        "input": {"location": "San Francisco"},
                    }
                ],
            },
        ],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assistant_msg = messages[1]
    assert "tool_calls" in assistant_msg
    assert assistant_msg["tool_calls"][0]["id"] == "tool_123"
    assert assistant_msg["tool_calls"][0]["name"] == "get_weather"


def test_tool_use_blocks_not_duplicated_in_content():
    """Tool use blocks must appear only in tool_calls, not also in content."""
    span = make_span()
    kwargs = {
        "model": "claude-3-opus-20240229",
        "messages": [
            {"role": "user", "content": "What's the weather in SF?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check that for you."},
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "get_weather",
                        "input": {"location": "San Francisco"},
                    },
                ],
            },
        ],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assistant_msg = messages[1]

    # tool_use block must be in tool_calls
    assert "tool_calls" in assistant_msg
    assert len(assistant_msg["tool_calls"]) == 1
    assert assistant_msg["tool_calls"][0]["id"] == "tool_123"

    # tool_use block must NOT appear in content
    content = assistant_msg.get("content", "")
    if isinstance(content, str):
        content_parsed = json.loads(content) if content.startswith("[") else content
    else:
        content_parsed = content
    if isinstance(content_parsed, list):
        types_in_content = [b.get("type") for b in content_parsed if isinstance(b, dict)]
        assert "tool_use" not in types_in_content, (
            "tool_use block must not be duplicated inside content"
        )


def test_tool_use_only_content_has_no_content_field():
    """When content is exclusively tool_use blocks, content should be None/absent."""
    span = make_span()
    kwargs = {
        "model": "claude-3-opus-20240229",
        "messages": [
            {"role": "user", "content": "What's the weather in SF?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "get_weather",
                        "input": {"location": "San Francisco"},
                    }
                ],
            },
        ],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assistant_msg = messages[1]

    assert "tool_calls" in assistant_msg
    # content should be None (no non-tool-use blocks)
    assert assistant_msg.get("content") is None


# ---------------------------------------------------------------------------
# Response / completion attribute tests
# ---------------------------------------------------------------------------

def _make_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(tool_id, name, input_data):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_data
    return block


def _make_response(content_blocks, stop_reason="end_turn", role="assistant"):
    usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    response = {
        "model": "claude-3-opus-20240229",
        "id": "msg_abc123",
        "role": role,
        "stop_reason": stop_reason,
        "content": content_blocks,
        "usage": usage,
    }
    return response


def test_output_messages_text_response():
    """gen_ai.output.messages should be a JSON array with the assistant response."""
    span = make_span()
    response = _make_response([_make_text_block("Why did the chicken cross the road?")])
    set_response_attributes(span, response)

    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in span.attributes, (
        "gen_ai.output.messages must be set"
    )
    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output) == 1
    assert output[0]["role"] == "assistant"
    assert output[0]["content"] == "Why did the chicken cross the road?"

    # Old attribute must NOT be set
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content" not in span.attributes
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role" not in span.attributes


def test_response_finish_reasons_attribute():
    """gen_ai.response.finish_reasons should be a list on the span."""
    span = make_span()
    response = _make_response([_make_text_block("Hello")], stop_reason="end_turn")
    set_response_attributes(span, response)

    assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS in span.attributes, (
        "gen_ai.response.finish_reasons must be set"
    )
    finish_reasons = span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]
    assert isinstance(finish_reasons, (list, tuple))
    assert "end_turn" in finish_reasons

    # Per-completion finish_reason must NOT be set as old attr
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.finish_reason" not in span.attributes


def test_finish_reasons_set_when_content_tracing_disabled():
    """gen_ai.response.finish_reasons must be recorded even when TRACELOOP_TRACE_CONTENT=false."""
    os.environ[TRACELOOP_TRACE_CONTENT] = "false"

    span = make_span()
    response = _make_response([_make_text_block("Secret content")], stop_reason="end_turn")
    set_response_attributes(span, response)

    assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS in span.attributes, (
        "gen_ai.response.finish_reasons must be set regardless of content tracing"
    )
    assert "end_turn" in span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]
    # Content must NOT be present
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


def test_streaming_finish_reasons_set_when_content_tracing_disabled():
    """Streaming finish_reasons must be recorded even when TRACELOOP_TRACE_CONTENT=false."""
    os.environ[TRACELOOP_TRACE_CONTENT] = "false"

    span = make_span()
    events = [{"type": "text", "text": "Secret content", "finish_reason": "end_turn", "index": 0}]
    set_streaming_response_attributes(span, events)

    assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS in span.attributes, (
        "streaming finish_reasons must be set regardless of content tracing"
    )
    assert "end_turn" in span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]
    # Content must NOT be present
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


def test_output_messages_tool_use_response():
    """Tool use in the response should appear as tool_calls in gen_ai.output.messages."""
    span = make_span()
    tool_block = _make_tool_use_block("tool_456", "get_weather", {"location": "NYC"})
    response = _make_response([tool_block], stop_reason="tool_use")
    set_response_attributes(span, response)

    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output) == 1
    assert "tool_calls" in output[0]
    tc = output[0]["tool_calls"][0]
    assert tc["id"] == "tool_456"
    assert tc["name"] == "get_weather"


def test_output_messages_streaming():
    """set_streaming_response_attributes should also use gen_ai.output.messages."""
    span = make_span()
    events = [
        {
            "type": "text",
            "text": "Streaming response",
            "finish_reason": "end_turn",
            "index": 0,
        }
    ]
    set_streaming_response_attributes(span, events)

    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in span.attributes, (
        "gen_ai.output.messages must be set for streaming"
    )
    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert output[0]["content"] == "Streaming response"

    assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS in span.attributes
    assert "end_turn" in span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]

    # Old attribute must NOT be set
    assert f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content" not in span.attributes


def test_output_messages_streaming_tool_use():
    """Streaming tool use should appear in gen_ai.output.messages."""
    span = make_span()
    events = [
        {
            "type": "tool_use",
            "id": "tool_789",
            "name": "get_weather",
            "input": '{"location": "Boston"}',
            "finish_reason": "tool_use",
            "index": 0,
        }
    ]
    set_streaming_response_attributes(span, events)

    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output) == 1
    assert "tool_calls" in output[0]
    assert output[0]["tool_calls"][0]["id"] == "tool_789"


# ---------------------------------------------------------------------------
# Span identity attribute tests (#4, #5)
# ---------------------------------------------------------------------------

def test_gen_ai_system_value_is_lowercase_anthropic():
    """gen_ai.system must use the spec enum value 'anthropic', not 'Anthropic'."""
    from opentelemetry.instrumentation.anthropic import _wrap
    from unittest.mock import patch, MagicMock

    tracer = MagicMock()
    captured = {}

    def fake_start_span(name, kind, attributes):
        captured["attributes"] = attributes
        span = MagicMock()
        span.is_recording.return_value = False
        return span

    tracer.start_span.side_effect = fake_start_span

    # Simulate _wrap being called for anthropic.chat
    to_wrap = {"span_name": "anthropic.chat"}
    wrapped_fn = MagicMock(return_value=None)

    with patch("opentelemetry.context.get_value", return_value=False):
        fn = _wrap(tracer, None, None, None, None, None, to_wrap)
        fn(wrapped_fn, MagicMock(), [], {"model": "claude-3-opus-20240229", "messages": [], "max_tokens": 10})

    actual = captured["attributes"].get(GenAIAttributes.GEN_AI_SYSTEM)
    assert actual == GenAiSystemValues.ANTHROPIC.value, (
        f"gen_ai.system must be 'anthropic' (lowercase), got '{actual}'"
    )


def test_gen_ai_provider_name_is_set():
    """gen_ai.provider.name must be set on every span."""
    from opentelemetry.instrumentation.anthropic import _wrap
    from unittest.mock import patch, MagicMock

    tracer = MagicMock()
    captured = {}

    def fake_start_span(name, kind, attributes):
        captured["attributes"] = attributes
        span = MagicMock()
        span.is_recording.return_value = False
        return span

    tracer.start_span.side_effect = fake_start_span

    to_wrap = {"span_name": "anthropic.chat"}
    wrapped_fn = MagicMock(return_value=None)

    with patch("opentelemetry.context.get_value", return_value=False):
        fn = _wrap(tracer, None, None, None, None, None, to_wrap)
        fn(wrapped_fn, MagicMock(), [], {"model": "claude-3-opus-20240229", "messages": [], "max_tokens": 10})

    assert GenAIAttributes.GEN_AI_PROVIDER_NAME in captured["attributes"], (
        "gen_ai.provider.name must be set on every span"
    )
    assert captured["attributes"][GenAIAttributes.GEN_AI_PROVIDER_NAME] == GenAiSystemValues.ANTHROPIC.value


def test_gen_ai_operation_name_chat():
    """gen_ai.operation.name must be 'chat' for Messages API spans."""
    from opentelemetry.instrumentation.anthropic import _wrap
    from unittest.mock import patch, MagicMock

    tracer = MagicMock()
    captured = {}

    def fake_start_span(name, kind, attributes):
        captured["attributes"] = attributes
        span = MagicMock()
        span.is_recording.return_value = False
        return span

    tracer.start_span.side_effect = fake_start_span

    to_wrap = {"span_name": "anthropic.chat"}
    wrapped_fn = MagicMock(return_value=None)

    with patch("opentelemetry.context.get_value", return_value=False):
        fn = _wrap(tracer, None, None, None, None, None, to_wrap)
        fn(wrapped_fn, MagicMock(), [], {"model": "claude-3-opus-20240229", "messages": [], "max_tokens": 10})

    assert GenAIAttributes.GEN_AI_OPERATION_NAME in captured["attributes"], (
        "gen_ai.operation.name must be set"
    )
    assert captured["attributes"][GenAIAttributes.GEN_AI_OPERATION_NAME] == GenAiOperationNameValues.CHAT.value


def test_gen_ai_operation_name_completion():
    """gen_ai.operation.name must be 'text_completion' for Completions API spans."""
    from opentelemetry.instrumentation.anthropic import _wrap
    from unittest.mock import patch, MagicMock

    tracer = MagicMock()
    captured = {}

    def fake_start_span(name, kind, attributes):
        captured["attributes"] = attributes
        span = MagicMock()
        span.is_recording.return_value = False
        return span

    tracer.start_span.side_effect = fake_start_span

    to_wrap = {"span_name": "anthropic.completion"}
    wrapped_fn = MagicMock(return_value=None)

    with patch("opentelemetry.context.get_value", return_value=False):
        fn = _wrap(tracer, None, None, None, None, None, to_wrap)
        fn(wrapped_fn, MagicMock(), [], {"model": "claude-2", "prompt": "Hello", "max_tokens_to_sample": 100})

    assert (
        captured["attributes"][GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAiOperationNameValues.TEXT_COMPLETION.value
    )


# ---------------------------------------------------------------------------
# Streaming tool_calls.arguments JSON serialization test (#6)
# ---------------------------------------------------------------------------

def test_streaming_tool_arguments_are_json_serialized():
    """Streaming tool_calls.arguments must be JSON-serialized like non-streaming paths."""
    span = make_span()
    events = [
        {
            "type": "tool_use",
            "id": "tool_abc",
            "name": "get_weather",
            "input": {"location": "Boston", "unit": "celsius"},
            "finish_reason": "tool_use",
            "index": 0,
        }
    ]
    set_streaming_response_attributes(span, events)

    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    tc = output[0]["tool_calls"][0]
    assert isinstance(tc["arguments"], str), (
        "tool_calls.arguments must be a JSON string, not a raw dict"
    )
    parsed = json.loads(tc["arguments"])
    assert parsed == {"location": "Boston", "unit": "celsius"}


def test_streaming_tool_arguments_not_double_encoded_when_input_is_string():
    """Streaming input arrives as an accumulated JSON string (from partial_json deltas).
    Calling json.dumps on it again would double-encode it — arguments must remain
    parseable as a plain JSON object, not a JSON-encoded string.
    """
    span = make_span()
    # input is a string here — exactly as streaming.py produces it after
    # accumulating input_json_delta fragments
    events = [
        {
            "type": "tool_use",
            "id": "tool_abc",
            "name": "get_weather",
            "input": '{"location": "Boston", "unit": "celsius"}',
            "finish_reason": "tool_use",
            "index": 0,
        }
    ]
    set_streaming_response_attributes(span, events)

    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    tc = output[0]["tool_calls"][0]
    assert isinstance(tc["arguments"], str), "arguments must be a string"
    parsed = json.loads(tc["arguments"])
    assert parsed == {"location": "Boston", "unit": "celsius"}, (
        "arguments must parse to the original dict, not a double-encoded string"
    )


# ---------------------------------------------------------------------------
# max_tokens fallback test (#2)
# ---------------------------------------------------------------------------

def test_max_tokens_set_for_messages_api():
    """gen_ai.request.max_tokens must be set from max_tokens (Messages API)."""
    span = make_span()
    kwargs = {
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 512,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    assert GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS in span.attributes, (
        "gen_ai.request.max_tokens must be set for Messages API calls"
    )
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 512


def test_max_tokens_set_for_completions_api():
    """gen_ai.request.max_tokens must be set from max_tokens_to_sample (legacy Completions API)."""
    span = make_span()
    kwargs = {
        "model": "claude-2",
        "prompt": "Hello",
        "max_tokens_to_sample": 256,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    assert GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS in span.attributes, (
        "gen_ai.request.max_tokens must be set for Completions API calls"
    )
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 256


def test_max_tokens_to_sample_takes_precedence_when_both_provided():
    """When both max_tokens_to_sample and max_tokens are present,
    max_tokens_to_sample (legacy) wins via `or` short-circuit.
    This documents the current behaviour so a future change would be intentional.
    """
    span = make_span()
    kwargs = {
        "model": "claude-2",
        "prompt": "Hello",
        "max_tokens_to_sample": 100,
        "max_tokens": 512,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 100, (
        "max_tokens_to_sample takes precedence over max_tokens when both are set"
    )


# ---------------------------------------------------------------------------
# Async finish_reasons with content tracing disabled (#2 async path)
# ---------------------------------------------------------------------------

def test_async_finish_reasons_set_when_content_tracing_disabled():
    """_aset_span_completions must record finish_reasons even when content tracing is off."""
    from opentelemetry.instrumentation.anthropic.span_utils import aset_response_attributes

    os.environ[TRACELOOP_TRACE_CONTENT] = "false"

    span = make_span()
    response = _make_response([_make_text_block("Secret content")], stop_reason="end_turn")
    asyncio.run(aset_response_attributes(span, response))

    assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS in span.attributes, (
        "async path: finish_reasons must be set regardless of content tracing"
    )
    assert "end_turn" in span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


# ---------------------------------------------------------------------------
# _awrap span identity attributes (#4 async path)
# ---------------------------------------------------------------------------

def test_awrap_gen_ai_provider_name_and_operation_name():
    """_awrap must set gen_ai.provider.name and gen_ai.operation.name same as _wrap."""
    import asyncio as _asyncio
    from opentelemetry.instrumentation.anthropic import _awrap
    from unittest.mock import patch, MagicMock, AsyncMock

    tracer = MagicMock()
    captured = {}

    def fake_start_span(name, kind, attributes):
        captured["attributes"] = attributes
        span = MagicMock()
        span.is_recording.return_value = False
        return span

    tracer.start_span.side_effect = fake_start_span

    to_wrap = {"span_name": "anthropic.chat"}
    wrapped_fn = AsyncMock(return_value=None)

    async def run():
        with patch("opentelemetry.context.get_value", return_value=False):
            fn = _awrap(tracer, None, None, None, None, None, to_wrap)
            await fn(wrapped_fn, MagicMock(), [], {"model": "claude-3-opus-20240229", "messages": [], "max_tokens": 10})

    _asyncio.run(run())

    assert captured["attributes"][GenAIAttributes.GEN_AI_PROVIDER_NAME] == GenAiSystemValues.ANTHROPIC.value
    assert captured["attributes"][GenAIAttributes.GEN_AI_OPERATION_NAME] == GenAiOperationNameValues.CHAT.value


# ---------------------------------------------------------------------------
# Multiple tool_use blocks in one input message
# ---------------------------------------------------------------------------

def test_multiple_tool_use_blocks_in_single_message():
    """Multiple tool_use blocks in one message must all appear in tool_calls,
    none in content, and content must be None.
    """
    span = make_span()
    kwargs = {
        "model": "claude-3-opus-20240229",
        "messages": [
            {"role": "user", "content": "What's the weather in Boston and NYC?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "get_weather",
                        "input": {"location": "Boston"},
                    },
                    {
                        "type": "tool_use",
                        "id": "tool_2",
                        "name": "get_weather",
                        "input": {"location": "New York"},
                    },
                ],
            },
        ],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assistant_msg = messages[1]

    assert len(assistant_msg["tool_calls"]) == 2
    ids = {tc["id"] for tc in assistant_msg["tool_calls"]}
    assert ids == {"tool_1", "tool_2"}
    assert assistant_msg.get("content") is None, (
        "content must be None when all blocks are tool_use"
    )
