"""
Unit tests for OTel GenAI semantic conventions compliance in span attributes.

These tests are written per the OTel GenAI spec JSON schemas:
  - gen-ai-input-messages.json:  messages use "parts" array with typed entries
  - gen-ai-output-messages.json: messages use "parts" array, finish_reason per message
  - Finish reasons enum: stop, length, content_filter, tool_call, error
  - Tool calls: {"type": "tool_call", "id": ..., "name": ..., "arguments": {...}} in parts
  - Reasoning:  {"type": "reasoning", "content": "..."} in parts
  - gen_ai.system_instructions (standalone attribute)
  - gen_ai.tool.definitions (JSON array)
  - gen_ai.response.finish_reasons (array of mapped OTel enum values)

Spec refs:
  https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
  https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-input-messages.json
  https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-output-messages.json
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
# Input attribute tests — messages must use "parts" structure per spec
# ---------------------------------------------------------------------------

def test_input_messages_simple_user_message():
    """gen_ai.input.messages must use parts structure per input-messages JSON schema."""
    span = make_span()
    kwargs = {
        "model": "claude-3-opus-20240229",
        "messages": [{"role": "user", "content": "Tell me a joke"}],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in span.attributes
    messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert messages == [
        {"role": "user", "parts": [{"type": "text", "content": "Tell me a joke"}]}
    ]


def test_input_messages_multi_turn():
    """gen_ai.input.messages should include all turns with parts structure."""
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
    assert messages[0] == {"role": "user", "parts": [{"type": "text", "content": "Hello"}]}
    assert messages[1] == {"role": "assistant", "parts": [{"type": "text", "content": "Hi there!"}]}
    assert messages[2] == {"role": "user", "parts": [{"type": "text", "content": "How are you?"}]}


def test_system_instructions_attribute():
    """gen_ai.system_instructions must be a JSON array of typed parts, NOT a plain string.

    OTel spec: gen_ai.system_instructions is a flat array of parts
    (NOT wrapped in {role, parts}).  Example:
        [{"type": "text", "content": "You are a helpful assistant."}]
    """
    span = make_span()
    kwargs = {
        "model": "claude-3-opus-20240229",
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS in span.attributes
    system_val = span.attributes[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS]
    # Must be valid JSON containing an array of parts
    parsed = json.loads(system_val)
    assert isinstance(parsed, list)
    assert parsed == [{"type": "text", "content": "You are a helpful assistant."}]

    # System should NOT appear as part of gen_ai.input.messages
    messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    roles = [m["role"] for m in messages]
    assert "system" not in roles


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

    assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS in span.attributes
    defs = json.loads(span.attributes[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS])
    assert len(defs) == 1
    assert defs[0]["name"] == "get_weather"
    assert defs[0]["description"] == "Get the current weather"
    assert "input_schema" in defs[0]


def test_input_messages_with_tool_calls_in_content():
    """Tool use blocks in assistant input messages should appear as tool_call parts."""
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
    assert assistant_msg["role"] == "assistant"
    # Tool calls should be tool_call parts
    tool_call_parts = [p for p in assistant_msg["parts"] if p["type"] == "tool_call"]
    assert len(tool_call_parts) == 1
    assert tool_call_parts[0]["id"] == "tool_123"
    assert tool_call_parts[0]["name"] == "get_weather"


def test_tool_use_blocks_not_duplicated_in_content():
    """Tool use blocks must appear as tool_call parts, text as text parts, no duplication."""
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

    # Should have both text and tool_call parts
    part_types = [p["type"] for p in assistant_msg["parts"]]
    assert "text" in part_types
    assert "tool_call" in part_types
    # No raw "tool_use" type (Anthropic native) should leak through
    assert "tool_use" not in part_types

    text_parts = [p for p in assistant_msg["parts"] if p["type"] == "text"]
    assert text_parts[0]["content"] == "Let me check that for you."

    tool_call_parts = [p for p in assistant_msg["parts"] if p["type"] == "tool_call"]
    assert len(tool_call_parts) == 1
    assert tool_call_parts[0]["id"] == "tool_123"


def test_tool_use_only_content_produces_only_tool_call_parts():
    """When content is exclusively tool_use blocks, parts should only contain tool_call entries."""
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

    # Only tool_call parts, no text parts
    assert all(p["type"] == "tool_call" for p in assistant_msg["parts"])
    assert len(assistant_msg["parts"]) == 1


# ---------------------------------------------------------------------------
# Response / completion attribute tests — parts structure + mapped finish_reasons
# ---------------------------------------------------------------------------

def _make_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_thinking_block(thinking_text):
    block = MagicMock()
    block.type = "thinking"
    block.thinking = thinking_text
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
    """gen_ai.output.messages must use parts structure with finish_reason per message."""
    span = make_span()
    response = _make_response([_make_text_block("Why did the chicken cross the road?")])
    set_response_attributes(span, response)

    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in span.attributes
    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output) == 1
    assert output[0]["role"] == "assistant"
    assert output[0]["parts"] == [{"type": "text", "content": "Why did the chicken cross the road?"}]
    assert output[0]["finish_reason"] == "stop"


def test_response_finish_reasons_mapped_to_otel_enum():
    """gen_ai.response.finish_reasons must map Anthropic values to OTel enum."""
    span = make_span()
    response = _make_response([_make_text_block("Hello")], stop_reason="end_turn")
    set_response_attributes(span, response)

    finish_reasons = span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]
    assert isinstance(finish_reasons, (list, tuple))
    # Anthropic "end_turn" must map to OTel "stop"
    assert finish_reasons == ["stop"]


def test_finish_reason_mapping_tool_use():
    """Anthropic 'tool_use' must map to OTel 'tool_call'."""
    span = make_span()
    tool_block = _make_tool_use_block("tool_1", "get_weather", {"location": "NYC"})
    response = _make_response([tool_block], stop_reason="tool_use")
    set_response_attributes(span, response)

    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ["tool_call"]


def test_finish_reason_mapping_max_tokens():
    """Anthropic 'max_tokens' must map to OTel 'length'."""
    span = make_span()
    response = _make_response([_make_text_block("truncated...")], stop_reason="max_tokens")
    set_response_attributes(span, response)

    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ["length"]


def test_finish_reasons_set_when_content_tracing_disabled():
    """gen_ai.response.finish_reasons must be recorded even when TRACELOOP_TRACE_CONTENT=false."""
    os.environ[TRACELOOP_TRACE_CONTENT] = "false"

    span = make_span()
    response = _make_response([_make_text_block("Secret content")], stop_reason="end_turn")
    set_response_attributes(span, response)

    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


def test_streaming_finish_reasons_set_when_content_tracing_disabled():
    """Streaming finish_reasons must be recorded even when TRACELOOP_TRACE_CONTENT=false."""
    os.environ[TRACELOOP_TRACE_CONTENT] = "false"

    span = make_span()
    events = [{"type": "text", "text": "Secret content", "finish_reason": "end_turn", "index": 0}]
    set_streaming_response_attributes(span, events)

    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


def test_finish_reason_empty_string_when_none():
    """finish_reason must be '' (not omitted) when stop_reason is None (Bedrock convention)."""
    span = make_span()
    response = _make_response([_make_text_block("Hello")], stop_reason=None)
    set_response_attributes(span, response)

    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output) == 1
    assert "finish_reason" in output[0], "finish_reason key must always be present"
    assert output[0]["finish_reason"] == "", (
        f"Expected '' for missing stop_reason, got '{output[0]['finish_reason']}'"
    )


def test_streaming_finish_reason_empty_string_when_none():
    """Streaming: finish_reason must be '' when no finish_reason in events."""
    span = make_span()
    events = [{"type": "text", "text": "Hello", "index": 0}]
    set_streaming_response_attributes(span, events)

    raw = span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
    if raw:
        output = json.loads(raw)
        assert output[0]["finish_reason"] == "", (
            f"Expected '' for missing streaming finish_reason, got '{output[0]['finish_reason']}'"
        )


def test_output_messages_tool_use_response():
    """Tool use in the response should appear as tool_call parts."""
    span = make_span()
    tool_block = _make_tool_use_block("tool_456", "get_weather", {"location": "NYC"})
    response = _make_response([tool_block], stop_reason="tool_use")
    set_response_attributes(span, response)

    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output) == 1
    assert output[0]["role"] == "assistant"
    assert output[0]["finish_reason"] == "tool_call"

    tool_call_parts = [p for p in output[0]["parts"] if p["type"] == "tool_call"]
    assert len(tool_call_parts) == 1
    assert tool_call_parts[0]["id"] == "tool_456"
    assert tool_call_parts[0]["name"] == "get_weather"
    assert tool_call_parts[0]["arguments"] == {"location": "NYC"}


def test_output_messages_text_and_tool_use_combined():
    """Text and tool_use in the same response must be parts in a single message."""
    span = make_span()
    text_block = _make_text_block("Let me check that.")
    tool_block = _make_tool_use_block("tool_456", "get_weather", {"location": "NYC"})
    response = _make_response([text_block, tool_block], stop_reason="tool_use")
    set_response_attributes(span, response)

    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output) == 1
    assert output[0]["role"] == "assistant"

    parts = output[0]["parts"]
    assert parts[0] == {"type": "text", "content": "Let me check that."}
    assert parts[1]["type"] == "tool_call"
    assert parts[1]["id"] == "tool_456"
    assert parts[1]["name"] == "get_weather"


def test_output_messages_thinking_as_reasoning_part():
    """Thinking content must be a ReasoningPart inside the assistant message's parts array."""
    span = make_span()
    thinking_block = _make_thinking_block("Let me think about this...")
    text_block = _make_text_block("The answer is 42.")
    response = _make_response([thinking_block, text_block], stop_reason="end_turn")
    set_response_attributes(span, response)

    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    # Must be a single assistant message, NOT separate "thinking" and "assistant" messages
    assert len(output) == 1
    assert output[0]["role"] == "assistant"
    assert output[0]["finish_reason"] == "stop"

    parts = output[0]["parts"]
    assert len(parts) == 2
    assert parts[0] == {"type": "reasoning", "content": "Let me think about this..."}
    assert parts[1] == {"type": "text", "content": "The answer is 42."}


def test_output_messages_streaming():
    """set_streaming_response_attributes should use parts structure."""
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

    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in span.attributes
    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output) == 1
    assert output[0]["role"] == "assistant"
    assert output[0]["parts"] == [{"type": "text", "content": "Streaming response"}]
    assert output[0]["finish_reason"] == "stop"

    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]


def test_output_messages_streaming_tool_use():
    """Streaming tool use should appear as tool_call parts."""
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
    assert output[0]["role"] == "assistant"
    assert output[0]["finish_reason"] == "tool_call"

    tool_call_parts = [p for p in output[0]["parts"] if p["type"] == "tool_call"]
    assert len(tool_call_parts) == 1
    assert tool_call_parts[0]["id"] == "tool_789"
    assert tool_call_parts[0]["name"] == "get_weather"
    assert tool_call_parts[0]["arguments"] == {"location": "Boston"}


def test_streaming_consolidated_single_message():
    """Streaming text + tool_use events must be consolidated into a single assistant message."""
    span = make_span()
    events = [
        {
            "type": "text",
            "text": "Let me check that.",
            "finish_reason": "tool_use",
            "index": 0,
        },
        {
            "type": "tool_use",
            "id": "tool_789",
            "name": "get_weather",
            "input": '{"location": "Boston"}',
            "finish_reason": "tool_use",
            "index": 1,
        },
    ]
    set_streaming_response_attributes(span, events)

    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    # Must be a single message, not separate messages for text and tool_call
    assert len(output) == 1
    assert output[0]["role"] == "assistant"
    assert output[0]["finish_reason"] == "tool_call"

    parts = output[0]["parts"]
    assert parts[0] == {"type": "text", "content": "Let me check that."}
    assert parts[1]["type"] == "tool_call"
    assert parts[1]["id"] == "tool_789"


def test_streaming_thinking_as_reasoning_part():
    """Streaming thinking events must become reasoning parts in a single assistant message."""
    span = make_span()
    events = [
        {
            "type": "thinking",
            "text": "Let me reason about this...",
            "finish_reason": "end_turn",
            "index": 0,
        },
        {
            "type": "text",
            "text": "The answer is 42.",
            "finish_reason": "end_turn",
            "index": 1,
        },
    ]
    set_streaming_response_attributes(span, events)

    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output) == 1
    assert output[0]["role"] == "assistant"
    assert output[0]["finish_reason"] == "stop"

    parts = output[0]["parts"]
    assert len(parts) == 2
    assert parts[0] == {"type": "reasoning", "content": "Let me reason about this..."}
    assert parts[1] == {"type": "text", "content": "The answer is 42."}


# ---------------------------------------------------------------------------
# Span identity attribute tests
# ---------------------------------------------------------------------------

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

    assert GenAIAttributes.GEN_AI_PROVIDER_NAME in captured["attributes"]
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
# Streaming tool_calls.arguments serialization tests
# ---------------------------------------------------------------------------

def test_streaming_tool_arguments_dict_input():
    """When streaming input is a dict, arguments must be serialized as an object in the part."""
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
    tc = [p for p in output[0]["parts"] if p["type"] == "tool_call"][0]
    assert tc["arguments"] == {"location": "Boston", "unit": "celsius"}


def test_streaming_tool_arguments_string_input():
    """When streaming input arrives as an accumulated JSON string, it must be parsed to an object."""
    span = make_span()
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
    tc = [p for p in output[0]["parts"] if p["type"] == "tool_call"][0]
    assert tc["arguments"] == {"location": "Boston", "unit": "celsius"}


# ---------------------------------------------------------------------------
# max_tokens fallback tests
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

    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 256


def test_max_tokens_to_sample_takes_precedence_when_both_provided():
    """When both max_tokens_to_sample and max_tokens are present,
    max_tokens_to_sample (legacy) wins via `or` short-circuit.
    """
    span = make_span()
    kwargs = {
        "model": "claude-2",
        "prompt": "Hello",
        "max_tokens_to_sample": 100,
        "max_tokens": 512,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 100


# ---------------------------------------------------------------------------
# Async finish_reasons with content tracing disabled
# ---------------------------------------------------------------------------

def test_async_finish_reasons_set_when_content_tracing_disabled():
    """_aset_span_completions must record mapped finish_reasons even when content tracing is off."""
    from opentelemetry.instrumentation.anthropic.span_utils import aset_response_attributes

    os.environ[TRACELOOP_TRACE_CONTENT] = "false"

    span = make_span()
    response = _make_response([_make_text_block("Secret content")], stop_reason="end_turn")
    asyncio.run(aset_response_attributes(span, response))

    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


# ---------------------------------------------------------------------------
# _awrap span identity attributes (async path)
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
    """Multiple tool_use blocks in one message must all appear as tool_call parts."""
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

    tool_call_parts = [p for p in assistant_msg["parts"] if p["type"] == "tool_call"]
    assert len(tool_call_parts) == 2
    ids = {tc["id"] for tc in tool_call_parts}
    assert ids == {"tool_1", "tool_2"}
    # No text parts should be present
    text_parts = [p for p in assistant_msg["parts"] if p["type"] == "text"]
    assert len(text_parts) == 0


# ---------------------------------------------------------------------------
# Missing coverage tests — expose semconv compliance gaps
# ---------------------------------------------------------------------------

def test_system_instructions_list_of_text_blocks():
    """When system is a list of text blocks, gen_ai.system_instructions must be
    a JSON array of TextParts, NOT a concatenated string.

    Anthropic allows system: [{type: "text", text: "..."},...].
    Spec: [{"type": "text", "content": "Part 1"}, {"type": "text", "content": "Part 2"}]
    """
    span = make_span()
    kwargs = {
        "model": "claude-3-opus-20240229",
        "system": [
            {"type": "text", "text": "You are a helpful assistant."},
            {"type": "text", "text": "Always be concise."},
        ],
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    system_val = span.attributes[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS]
    parsed = json.loads(system_val)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0] == {"type": "text", "content": "You are a helpful assistant."}
    assert parsed[1] == {"type": "text", "content": "Always be concise."}


def test_finish_reason_none_not_set():
    """When stop_reason is None, gen_ai.response.finish_reasons must NOT be set."""
    span = make_span()
    response = _make_response([_make_text_block("Hello")], stop_reason=None)
    set_response_attributes(span, response)

    assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in span.attributes


def test_finish_reason_stop_sequence_mapping():
    """Anthropic 'stop_sequence' must map to OTel 'stop'."""
    span = make_span()
    response = _make_response(
        [_make_text_block("partial...")], stop_reason="stop_sequence"
    )
    set_response_attributes(span, response)

    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ["stop"]


def test_tool_result_in_input_as_tool_call_response():
    """tool_result blocks in input must be mapped to tool_call_response parts."""
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
                        "id": "call_123",
                        "name": "get_weather",
                        "input": {"location": "San Francisco"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "72°F, sunny",
                    }
                ],
            },
        ],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    tool_result_msg = messages[2]
    assert tool_result_msg["role"] == "user"
    assert len(tool_result_msg["parts"]) == 1
    assert tool_result_msg["parts"][0] == {
        "type": "tool_call_response",
        "id": "call_123",
        "response": "72°F, sunny",
    }
    # No raw "tool_result" type should leak through
    assert "tool_result" not in [p["type"] for p in tool_result_msg["parts"]]


def test_image_with_upload_produces_uri_part():
    """Uploaded base64 images must use OTel UriPart format, NOT OpenAI image_url."""
    from opentelemetry.instrumentation.anthropic.config import Config

    original = Config.upload_base64_image

    async def mock_upload(*args):
        return "https://example.com/uploaded.jpg"

    Config.upload_base64_image = mock_upload

    try:
        span = make_span()
        kwargs = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgoAAAANSUhEUg==",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
        }
        asyncio.run(aset_input_attributes(span, kwargs))

        messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
        image_part = messages[0]["parts"][1]
        # OTel spec: UriPart
        assert image_part == {
            "type": "uri",
            "modality": "image",
            "uri": "https://example.com/uploaded.jpg",
        }
    finally:
        Config.upload_base64_image = original


def test_image_without_upload_produces_blob_part():
    """Non-uploaded base64 images must use OTel BlobPart, NOT raw Anthropic format."""
    from opentelemetry.instrumentation.anthropic.config import Config

    original = Config.upload_base64_image
    Config.upload_base64_image = None

    try:
        span = make_span()
        kwargs = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgoAAAANSUhEUg==",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
        }
        asyncio.run(aset_input_attributes(span, kwargs))

        messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
        image_part = messages[0]["parts"][1]
        # OTel spec: BlobPart
        assert image_part == {
            "type": "blob",
            "modality": "image",
            "mime_type": "image/png",
            "content": "iVBORw0KGgoAAAANSUhEUg==",
        }
    finally:
        Config.upload_base64_image = original


def test_streaming_finish_reason_null_omitted_from_json():
    """When no finish_reason is available, the key must be present with empty
    string value — NOT serialized as null, NOT omitted (Bedrock convention)."""
    span = make_span()
    # Event with no finish_reason key at all
    events = [{"type": "text", "text": "Hello world", "index": 0}]
    set_streaming_response_attributes(span, events)

    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output) == 1
    assert output[0]["role"] == "assistant"
    assert output[0]["parts"] == [{"type": "text", "content": "Hello world"}]
    # finish_reason key must be present with empty string fallback
    assert output[0]["finish_reason"] == ""


def test_streaming_finish_reason_none_does_not_set_span_attr():
    """When streaming events have no finish_reason, gen_ai.response.finish_reasons
    must NOT be set on the span."""
    span = make_span()
    events = [{"type": "text", "text": "Hello", "index": 0}]
    set_streaming_response_attributes(span, events)

    assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in span.attributes


# ---------------------------------------------------------------------------
# P2 #1 — Streaming: unknown block types (e.g. redacted_thinking) must be skipped
# ---------------------------------------------------------------------------

def test_streaming_redacted_thinking_skipped():
    """redacted_thinking blocks must NOT produce a text part with content: None.

    Anthropic returns redacted_thinking blocks during extended thinking.
    These have no 'text' key.  The spec has no part type for them, so they
    must be silently dropped — not emitted as {"type": "text", "content": None}.
    """
    span = make_span()
    events = [
        {"type": "thinking", "text": "Let me reason...", "index": 0, "finish_reason": "end_turn"},
        {"type": "redacted_thinking", "index": 1, "finish_reason": "end_turn"},
        {"type": "text", "text": "The answer is 42.", "index": 2, "finish_reason": "end_turn"},
    ]
    set_streaming_response_attributes(span, events)

    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    parts = output[0]["parts"]

    # Only reasoning + text parts — redacted_thinking must be absent
    assert len(parts) == 2
    assert parts[0] == {"type": "reasoning", "content": "Let me reason..."}
    assert parts[1] == {"type": "text", "content": "The answer is 42."}

    # No part should have content == None
    for part in parts:
        if "content" in part:
            assert part["content"] is not None, f"Part has content: None — {part}"


def test_streaming_unknown_block_type_skipped():
    """Any unknown block type (not text, thinking, tool_use) must be skipped,
    not emitted as a text part with content: None."""
    span = make_span()
    events = [
        {"type": "text", "text": "Hello", "index": 0, "finish_reason": "end_turn"},
        {"type": "some_future_block_type", "index": 1, "finish_reason": "end_turn"},
    ]
    set_streaming_response_attributes(span, events)

    output = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    parts = output[0]["parts"]

    # Only the text part — unknown type must be dropped
    assert len(parts) == 1
    assert parts[0] == {"type": "text", "content": "Hello"}


# ---------------------------------------------------------------------------
# P2 #2 — Cache token constants must resolve to valid SpanAttributes
# ---------------------------------------------------------------------------

def test_set_token_usage_writes_dotted_cache_attribute_keys():
    """_set_token_usage must write the dotted cache attribute keys
    (gen_ai.usage.cache_read.input_tokens), NOT the old underscore-separated
    keys (gen_ai.usage.cache_read_input_tokens)."""
    from opentelemetry.instrumentation.anthropic.streaming import _set_token_usage

    span = make_span()
    complete_response = {
        "model": "claude-3-opus-20240229",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 20,
        },
        "events": [],
    }
    _set_token_usage(span, complete_response, 100, 50)

    # Dotted keys (correct, post-v0.5.0)
    assert span.attributes.get("gen_ai.usage.cache_read.input_tokens") == 30, \
        f"Expected dotted cache_read key; got attrs: {span.attributes}"
    assert span.attributes.get("gen_ai.usage.cache_creation.input_tokens") == 20, \
        f"Expected dotted cache_creation key; got attrs: {span.attributes}"

    # Old underscore-separated keys must NOT be written
    assert "gen_ai.usage.cache_read_input_tokens" not in span.attributes, \
        "Old underscore-separated cache_read key should not be written"
    assert "gen_ai.usage.cache_creation_input_tokens" not in span.attributes, \
        "Old underscore-separated cache_creation key should not be written"


# ---------------------------------------------------------------------------
# P3 #3 — _handle_completion must not double-record metrics, and must work
#          even without a duration_histogram
# ---------------------------------------------------------------------------

def _make_anthropic_stream(span, duration_histogram=None, event_logger=None):
    """Helper to create an AnthropicStream for unit testing.

    ObjectProxy (wrapt) needs a wrapped object that supports attribute
    assignment, so we wrap a MagicMock rather than a bare iterator.
    """
    from opentelemetry.instrumentation.anthropic.streaming import AnthropicStream

    wrapped = MagicMock()
    wrapped.__iter__ = MagicMock(return_value=iter([]))
    wrapped.__next__ = MagicMock(side_effect=StopIteration)

    stream = AnthropicStream(
        span=span,
        response=wrapped,
        instance=MagicMock(),
        start_time=0,
        duration_histogram=duration_histogram,
        event_logger=event_logger,
        kwargs={},
    )
    return stream


def test_handle_completion_without_duration_histogram():
    """_handle_completion must complete instrumentation (set response attrs,
    end span) even when no duration_histogram is configured.

    Current bug: all post-histogram logic is nested inside
    `if self._duration_histogram:`, so without it nothing happens.
    """
    from unittest.mock import patch

    span = make_span()
    span.is_recording.return_value = True
    span.end = MagicMock()

    stream = _make_anthropic_stream(span, duration_histogram=None)
    stream._complete_response = {
        "events": [{"type": "text", "text": "Hello", "index": 0, "finish_reason": "end_turn"}],
        "model": "claude-3-opus-20240229",
        "usage": {},
        "id": "msg_123",
    }

    with patch("opentelemetry.instrumentation.anthropic.streaming.Config") as mock_config:
        mock_config.enrich_token_usage = False
        stream._handle_completion()

    # Span must be ended even without a histogram
    span.end.assert_called_once()
    assert stream._instrumentation_completed is True


def test_handle_completion_records_duration_exactly_once():
    """duration_histogram.record must be called exactly once, not twice."""
    from unittest.mock import patch

    span = make_span()
    span.is_recording.return_value = True
    span.end = MagicMock()

    duration_histogram = MagicMock()

    stream = _make_anthropic_stream(span, duration_histogram=duration_histogram)
    stream._complete_response = {
        "events": [{"type": "text", "text": "Hi", "index": 0, "finish_reason": "end_turn"}],
        "model": "claude-3-opus-20240229",
        "usage": {},
        "id": "msg_456",
    }

    with patch("opentelemetry.instrumentation.anthropic.streaming.Config") as mock_config:
        mock_config.enrich_token_usage = False
        stream._handle_completion()

    # duration_histogram.record must be called exactly once
    assert duration_histogram.record.call_count == 1, \
        f"Expected 1 call to duration_histogram.record, got {duration_histogram.record.call_count}"


# ---------------------------------------------------------------------------
# P2 #2 — _content_to_parts must map thinking blocks to ReasoningPart
# ---------------------------------------------------------------------------

def test_input_thinking_block_mapped_to_reasoning_part():
    """In multi-turn extended-thinking conversations, assistant messages can
    contain type: 'thinking' blocks (echoed from previous turn).  These must
    be mapped to {"type": "reasoning", "content": "..."} — NOT fall through
    to the else clause that emits raw Anthropic dicts."""
    span = make_span()
    kwargs = {
        "model": "claude-3-7-sonnet-20250219",
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me calculate..."},
                    {"type": "text", "text": "The answer is 4."},
                ],
            },
            {"role": "user", "content": "And 3+3?"},
        ],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assistant_msg = messages[1]
    assert assistant_msg["role"] == "assistant"

    parts = assistant_msg["parts"]
    assert len(parts) == 2
    assert parts[0] == {"type": "reasoning", "content": "Let me calculate..."}
    assert parts[1] == {"type": "text", "content": "The answer is 4."}
    # No raw "thinking" type should leak through
    assert "thinking" not in [p["type"] for p in parts]


# ---------------------------------------------------------------------------
# P2 #3 — URL-referenced images must be handled in _content_to_parts
#          and _dump_system_content
# ---------------------------------------------------------------------------

def test_url_image_in_input_produces_uri_part():
    """Anthropic source.type == 'url' images must produce OTel UriPart,
    NOT fall through to the raw else clause."""
    span = make_span()
    kwargs = {
        "model": "claude-3-opus-20240229",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://example.com/photo.jpg",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    image_part = messages[0]["parts"][1]
    assert image_part == {
        "type": "uri",
        "modality": "image",
        "uri": "https://example.com/photo.jpg",
    }


def test_url_image_in_system_produces_uri_part():
    """URL images in system instructions must also produce OTel UriPart."""
    span = make_span()
    kwargs = {
        "model": "claude-3-opus-20240229",
        "system": [
            {"type": "text", "text": "You are a helpful assistant."},
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/logo.png",
                },
            },
        ],
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    system_val = json.loads(span.attributes[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS])
    assert len(system_val) == 2
    assert system_val[0] == {"type": "text", "content": "You are a helpful assistant."}
    assert system_val[1] == {
        "type": "uri",
        "modality": "image",
        "uri": "https://example.com/logo.png",
    }


def test_file_image_falls_through_gracefully():
    """Unknown source types (e.g. file) should fall through to raw dict."""
    span = make_span()
    kwargs = {
        "model": "claude-3-opus-20240229",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image",
                        "source": {
                            "type": "file",
                            "file_id": "file_abc123",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    image_part = messages[0]["parts"][1]
    # Unknown source type falls through — raw dict preserved
    assert image_part["type"] == "image"
    assert image_part["source"]["type"] == "file"


# ---------------------------------------------------------------------------
# P2 #4 — Streaming metric choice_counter must use mapped finish_reason
# ---------------------------------------------------------------------------

def test_streaming_metric_choice_counter_uses_mapped_finish_reason():
    """The choice_counter in _set_token_usage must apply _map_finish_reason
    to the raw Anthropic stop_reason, not emit it verbatim."""
    from opentelemetry.instrumentation.anthropic.streaming import _set_token_usage
    from opentelemetry.semconv_ai import SpanAttributes as SA

    span = make_span()
    choice_counter = MagicMock()

    complete_response = {
        "model": "claude-3-opus-20240229",
        "usage": {"input_tokens": 100, "output_tokens": 50},
        "events": [
            {"type": "text", "text": "Hello", "finish_reason": "end_turn"},
        ],
    }
    _set_token_usage(span, complete_response, 100, 50, {}, None, choice_counter)

    # choice_counter.add must have been called with mapped "stop", not raw "end_turn"
    assert choice_counter.add.call_count == 1
    call_attrs = choice_counter.add.call_args[1]["attributes"]
    assert call_attrs[SA.GEN_AI_RESPONSE_FINISH_REASON] == "stop", \
        f"Expected mapped 'stop', got '{call_attrs[SA.GEN_AI_RESPONSE_FINISH_REASON]}'"


# ---------------------------------------------------------------------------
# Finding 1 — Non-streaming _set_token_usage / _aset_token_usage in __init__
#              must use GEN_AI_RESPONSE_FINISH_REASON (not STOP_REASON)
#              and apply _map_finish_reason()
# ---------------------------------------------------------------------------

def test_nonstreaming_sync_choice_counter_uses_mapped_finish_reason():
    """_set_token_usage in __init__.py must use GEN_AI_RESPONSE_FINISH_REASON
    with a mapped OTel value, not GEN_AI_RESPONSE_STOP_REASON with raw Anthropic value."""
    from opentelemetry.instrumentation.anthropic import _set_token_usage
    from opentelemetry.semconv_ai import SpanAttributes as SA

    span = make_span()
    choice_counter = MagicMock()

    # Mock Anthropic response object
    usage = SimpleNamespace(input_tokens=100, output_tokens=50,
                            cache_read_input_tokens=0, cache_creation_input_tokens=0)
    response = SimpleNamespace(
        usage=usage,
        content=[SimpleNamespace(type="text", text="Hello")],
        completion=None,
        stop_reason="end_turn",
        model="claude-3-opus-20240229",
    )

    # Mock anthropic client (no count_tokens needed since usage is present)
    anthropic_client = MagicMock()

    _set_token_usage(span, anthropic_client, {}, response, {}, None, choice_counter)

    assert choice_counter.add.call_count == 1
    call_attrs = choice_counter.add.call_args[1]["attributes"]

    # Must use GEN_AI_RESPONSE_FINISH_REASON, NOT GEN_AI_RESPONSE_STOP_REASON
    assert SA.GEN_AI_RESPONSE_FINISH_REASON in call_attrs, \
        f"Expected GEN_AI_RESPONSE_FINISH_REASON in attrs, got keys: {list(call_attrs.keys())}"
    assert SA.GEN_AI_RESPONSE_STOP_REASON not in call_attrs, \
        "GEN_AI_RESPONSE_STOP_REASON should not be used — use GEN_AI_RESPONSE_FINISH_REASON"

    # Must be mapped: end_turn → stop
    assert call_attrs[SA.GEN_AI_RESPONSE_FINISH_REASON] == "stop", \
        f"Expected mapped 'stop', got '{call_attrs[SA.GEN_AI_RESPONSE_FINISH_REASON]}'"


def test_nonstreaming_async_choice_counter_uses_mapped_finish_reason():
    """_aset_token_usage in __init__.py must use GEN_AI_RESPONSE_FINISH_REASON
    with a mapped OTel value."""
    from opentelemetry.instrumentation.anthropic import _aset_token_usage
    from opentelemetry.semconv_ai import SpanAttributes as SA

    span = make_span()
    choice_counter = MagicMock()

    usage = SimpleNamespace(input_tokens=100, output_tokens=50,
                            cache_read_input_tokens=0, cache_creation_input_tokens=0)
    response = SimpleNamespace(
        usage=usage,
        content=[SimpleNamespace(type="text", text="Hello")],
        completion=None,
        stop_reason="tool_use",
        model="claude-3-opus-20240229",
    )

    anthropic_client = MagicMock()

    asyncio.run(_aset_token_usage(span, anthropic_client, {}, response, {}, None, choice_counter))

    assert choice_counter.add.call_count == 1
    call_attrs = choice_counter.add.call_args[1]["attributes"]

    assert SA.GEN_AI_RESPONSE_FINISH_REASON in call_attrs
    assert SA.GEN_AI_RESPONSE_STOP_REASON not in call_attrs
    # tool_use → tool_call
    assert call_attrs[SA.GEN_AI_RESPONSE_FINISH_REASON] == "tool_call", \
        f"Expected mapped 'tool_call', got '{call_attrs[SA.GEN_AI_RESPONSE_FINISH_REASON]}'"


# ---------------------------------------------------------------------------
# Finding 2 — redacted_thinking blocks in input must be skipped,
#              not fall through to raw dict (checklist §1: no raw provider blocks)
# ---------------------------------------------------------------------------

def test_input_redacted_thinking_block_skipped():
    """redacted_thinking blocks in input content should be skipped,
    not emitted as raw Anthropic dicts."""
    span = make_span()
    kwargs = {
        "model": "claude-3-7-sonnet-20250219",
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "redacted_thinking", "data": "ENCRYPTED_DATA"},
                    {"type": "text", "text": "The answer is 4."},
                ],
            },
            {"role": "user", "content": "Thanks!"},
        ],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assistant_msg = messages[1]
    parts = assistant_msg["parts"]

    # Should have 2 parts: reasoning + text (redacted_thinking skipped)
    assert len(parts) == 2, f"Expected 2 parts, got {len(parts)}: {parts}"
    assert parts[0] == {"type": "reasoning", "content": "Let me think..."}
    assert parts[1] == {"type": "text", "content": "The answer is 4."}
    # No raw Anthropic types should leak through
    part_types = [p["type"] for p in parts]
    assert "redacted_thinking" not in part_types, \
        "redacted_thinking must be skipped, not emitted as raw dict"


# ---------------------------------------------------------------------------
# §6 — Metric and event attributes must use GEN_AI_PROVIDER_NAME,
#       not deprecated GEN_AI_SYSTEM
# ---------------------------------------------------------------------------

def test_shared_metrics_attributes_uses_provider_name_not_system():
    """shared_metrics_attributes must use gen_ai.provider.name, not deprecated gen_ai.system."""
    from opentelemetry.instrumentation.anthropic.utils import shared_metrics_attributes

    response = SimpleNamespace(model="claude-3-opus-20240229")
    attrs = shared_metrics_attributes(response)

    assert GenAIAttributes.GEN_AI_PROVIDER_NAME in attrs, \
        f"Expected GEN_AI_PROVIDER_NAME in metric attrs, got keys: {list(attrs.keys())}"
    assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "anthropic"
    # Deprecated key must NOT be present
    assert GenAIAttributes.GEN_AI_SYSTEM not in attrs, \
        "Deprecated GEN_AI_SYSTEM should not be in metric attributes"


def test_error_metrics_attributes_uses_provider_name_not_system():
    """error_metrics_attributes must use gen_ai.provider.name, not deprecated gen_ai.system."""
    from opentelemetry.instrumentation.anthropic.utils import error_metrics_attributes

    attrs = error_metrics_attributes(ValueError("test"))

    assert GenAIAttributes.GEN_AI_PROVIDER_NAME in attrs, \
        f"Expected GEN_AI_PROVIDER_NAME in error metric attrs, got keys: {list(attrs.keys())}"
    assert attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "anthropic"
    assert GenAIAttributes.GEN_AI_SYSTEM not in attrs, \
        "Deprecated GEN_AI_SYSTEM should not be in error metric attributes"


def test_event_attributes_uses_provider_name_not_system():
    """EVENT_ATTRIBUTES in event_emitter must use gen_ai.provider.name."""
    from opentelemetry.instrumentation.anthropic.event_emitter import EVENT_ATTRIBUTES

    assert GenAIAttributes.GEN_AI_PROVIDER_NAME in EVENT_ATTRIBUTES, \
        f"Expected GEN_AI_PROVIDER_NAME in EVENT_ATTRIBUTES, got keys: {list(EVENT_ATTRIBUTES.keys())}"
    assert EVENT_ATTRIBUTES[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "anthropic"
    assert GenAIAttributes.GEN_AI_SYSTEM not in EVENT_ATTRIBUTES, \
        "Deprecated GEN_AI_SYSTEM should not be in EVENT_ATTRIBUTES"


# ---------------------------------------------------------------------------
# Regression: streaming must set token usage from API data even when
# enrich_token_usage is False (the default). Fixes #3949.
# ---------------------------------------------------------------------------

def _make_anthropic_async_stream(span, duration_histogram=None, event_logger=None):
    """Helper to create an AnthropicAsyncStream for unit testing."""
    from opentelemetry.instrumentation.anthropic.streaming import AnthropicAsyncStream

    wrapped = MagicMock()
    wrapped.__aiter__ = MagicMock(return_value=iter([]))
    wrapped.__anext__ = MagicMock(side_effect=StopAsyncIteration)

    stream = AnthropicAsyncStream(
        span=span,
        response=wrapped,
        instance=MagicMock(),
        start_time=0,
        duration_histogram=duration_histogram,
        event_logger=event_logger,
        kwargs={},
    )
    return stream


def test_streaming_sets_token_usage_from_api_without_enrich_flag():
    """When enrich_token_usage is False but the stream carries real usage
    data from the API, the span must still get gen_ai.usage.input_tokens
    and gen_ai.usage.output_tokens set.  Previously, the entire token-usage
    block was gated behind `if Config.enrich_token_usage`, so with the
    default (False) no token attributes were recorded on streaming spans."""
    from unittest.mock import patch

    span = make_span()
    span.is_recording.return_value = True
    span.end = MagicMock()

    stream = _make_anthropic_stream(span)
    stream._complete_response = {
        "events": [{"type": "text", "text": "Blue", "index": 0, "finish_reason": "end_turn"}],
        "model": "claude-haiku-4-5-20251001",
        "usage": {
            "input_tokens": 343,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        "id": "msg_streaming_test",
    }

    with patch("opentelemetry.instrumentation.anthropic.streaming.Config") as mock_config:
        mock_config.enrich_token_usage = False
        stream._handle_completion()

    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 343, \
        f"Expected input_tokens=343; got {span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)}"
    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 5, \
        f"Expected output_tokens=5; got {span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)}"
    assert stream._instrumentation_completed is True
    span.end.assert_called_once()


def test_streaming_skips_token_usage_without_api_data_and_enrich_disabled():
    """When enrich_token_usage is False AND the stream has no usage data,
    no token attributes should be set (no fallback estimation)."""
    from unittest.mock import patch

    span = make_span()
    span.is_recording.return_value = True
    span.end = MagicMock()

    stream = _make_anthropic_stream(span)
    stream._complete_response = {
        "events": [{"type": "text", "text": "Hello", "index": 0, "finish_reason": "end_turn"}],
        "model": "claude-3-opus-20240229",
        "usage": {},
        "id": "msg_no_usage",
    }

    with patch("opentelemetry.instrumentation.anthropic.streaming.Config") as mock_config:
        mock_config.enrich_token_usage = False
        stream._handle_completion()

    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS not in span.attributes, \
        "Token attributes should not be set when no API usage and enrich disabled"
    assert stream._instrumentation_completed is True
    span.end.assert_called_once()


def test_async_streaming_sets_token_usage_from_api_without_enrich_flag():
    """Async counterpart: AnthropicAsyncStream._complete_instrumentation must
    set token attributes from API usage even when enrich_token_usage is False."""
    from unittest.mock import patch

    span = make_span()
    span.is_recording.return_value = True
    span.end = MagicMock()

    stream = _make_anthropic_async_stream(span)
    stream._complete_response = {
        "events": [{"type": "text", "text": "Blue", "index": 0, "finish_reason": "end_turn"}],
        "model": "claude-haiku-4-5-20251001",
        "usage": {
            "input_tokens": 343,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        "id": "msg_async_streaming_test",
    }

    with patch("opentelemetry.instrumentation.anthropic.streaming.Config") as mock_config:
        mock_config.enrich_token_usage = False
        stream._complete_instrumentation()

    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 343, \
        f"Expected input_tokens=343; got {span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)}"
    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 5, \
        f"Expected output_tokens=5; got {span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)}"
    assert stream._instrumentation_completed is True
    span.end.assert_called_once()


def test_async_streaming_skips_token_usage_without_api_data_and_enrich_disabled():
    """Async counterpart: no token attributes when usage is empty and
    enrich_token_usage is False."""
    from unittest.mock import patch

    span = make_span()
    span.is_recording.return_value = True
    span.end = MagicMock()

    stream = _make_anthropic_async_stream(span)
    stream._complete_response = {
        "events": [{"type": "text", "text": "Hello", "index": 0, "finish_reason": "end_turn"}],
        "model": "claude-3-opus-20240229",
        "usage": {},
        "id": "msg_async_no_usage",
    }

    with patch("opentelemetry.instrumentation.anthropic.streaming.Config") as mock_config:
        mock_config.enrich_token_usage = False
        stream._complete_instrumentation()

    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS not in span.attributes, \
        "Token attributes should not be set when no API usage and enrich disabled"
    assert stream._instrumentation_completed is True
    span.end.assert_called_once()
# _map_finish_reason must return "" for falsy input, mapped value for known
# reasons, and the original string as-is for unknown reasons.
# ---------------------------------------------------------------------------

class TestMapFinishReason:
    from opentelemetry.instrumentation.anthropic.span_utils import _map_finish_reason
    _map_finish_reason = staticmethod(_map_finish_reason)

    @pytest.mark.parametrize("falsy_input", [None, "", 0, False])
    def test_returns_empty_string_for_falsy(self, falsy_input):
        assert self._map_finish_reason(falsy_input) == ""

    def test_maps_end_turn_to_stop(self):
        assert self._map_finish_reason("end_turn") == "stop"

    def test_maps_tool_use_to_tool_call(self):
        assert self._map_finish_reason("tool_use") == "tool_call"

    def test_maps_max_tokens_to_length(self):
        assert self._map_finish_reason("max_tokens") == "length"

    def test_maps_stop_sequence_to_stop(self):
        assert self._map_finish_reason("stop_sequence") == "stop"

    def test_passes_through_unknown_reason(self):
        assert self._map_finish_reason("some_new_reason") == "some_new_reason"
