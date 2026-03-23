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
    """System prompt should be in gen_ai.system_instructions, NOT as a message."""
    span = make_span()
    kwargs = {
        "model": "claude-3-opus-20240229",
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1024,
    }
    asyncio.run(aset_input_attributes(span, kwargs))

    assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS in span.attributes
    assert span.attributes[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS] == (
        "You are a helpful assistant."
    )

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
