"""Unit tests for gen_ai.input.messages / gen_ai.output.messages formatting.

These test the _set_input_messages and _set_output_messages code paths
using mocked spans, without VCR cassettes.
"""

import json
from unittest.mock import MagicMock

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    _set_input_messages,
    _set_output_messages,
    _map_finish_reason,
)
from opentelemetry.instrumentation.openai.shared.completion_wrappers import (
    _set_input_messages as _set_completion_input_messages,
)
from opentelemetry.instrumentation.openai.shared import (
    _get_vendor_from_url,
    _set_request_attributes,
    _set_response_attributes,
    metric_shared_attributes,
)


@pytest.fixture
def mock_span():
    span = MagicMock()
    span.is_recording.return_value = True
    attrs = {}

    def set_attribute(name, value):
        attrs[name] = value

    span.set_attribute = set_attribute
    span._attrs = attrs
    return span


def _get_input_messages(span):
    return json.loads(span._attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES])


def _get_output_messages(span):
    return json.loads(span._attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])


# ---------------------------------------------------------------------------
# Multi-turn input messages (system + user + assistant with tool_calls + tool)
# ---------------------------------------------------------------------------

class TestSetInputMessages:
    def test_multiturn_with_tool_calls(self, mock_span):
        messages = [
            {"role": "system", "content": "You are a weather assistant."},
            {"role": "user", "content": "What is the weather in NYC?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": '{"temp": 72, "unit": "F"}',
            },
        ]

        _set_input_messages(mock_span, messages)
        result = _get_input_messages(mock_span)

        assert len(result) == 4

        # system
        assert result[0]["role"] == "system"
        assert result[0]["parts"][0]["type"] == "text"
        assert result[0]["parts"][0]["content"] == "You are a weather assistant."

        # user
        assert result[1]["role"] == "user"
        assert result[1]["parts"][0]["content"] == "What is the weather in NYC?"

        # assistant with tool call (content=None → empty parts + tool_call_request)
        assert result[2]["role"] == "assistant"
        assert any(p["type"] == "tool_call" for p in result[2]["parts"])
        tool_part = next(
            p for p in result[2]["parts"] if p["type"] == "tool_call"
        )
        assert tool_part["name"] == "get_weather"
        assert tool_part["id"] == "call_abc123"
        assert tool_part["arguments"] == {"location": "NYC"}

        # tool response
        assert result[3]["role"] == "tool"
        assert result[3]["parts"][0]["type"] == "tool_call_response"
        assert result[3]["parts"][0]["id"] == "call_abc123"
        assert result[3]["parts"][0]["response"] == '{"temp": 72, "unit": "F"}'

    def test_user_with_none_content(self, mock_span):
        """User message with None content should produce empty parts list."""
        _set_input_messages(mock_span, [{"role": "user", "content": None}])
        result = _get_input_messages(mock_span)

        assert len(result) == 1
        assert result[0]["parts"] == []

    def test_system_with_none_content(self, mock_span):
        """System message with None content should produce empty parts list."""
        _set_input_messages(mock_span, [{"role": "system", "content": None}])
        result = _get_input_messages(mock_span)

        assert len(result) == 1
        assert result[0]["parts"] == []

    def test_developer_role(self, mock_span):
        _set_input_messages(mock_span, [{"role": "developer", "content": "Be concise."}])
        result = _get_input_messages(mock_span)

        assert result[0]["role"] == "developer"
        assert result[0]["parts"][0]["content"] == "Be concise."

    def test_none_messages(self, mock_span):
        """None messages input should be a no-op."""
        _set_input_messages(mock_span, None)
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in mock_span._attrs

    def test_assistant_with_text_and_tool_calls(self, mock_span):
        """Assistant message with both text content and tool calls."""
        _set_input_messages(mock_span, [
            {
                "role": "assistant",
                "content": "Let me check that for you.",
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            }
        ])
        result = _get_input_messages(mock_span)
        parts = result[0]["parts"]

        text_parts = [p for p in parts if p.get("type") == "text"]
        tool_parts = [p for p in parts if p.get("type") == "tool_call"]
        assert len(text_parts) == 1
        assert text_parts[0]["content"] == "Let me check that for you."
        assert len(tool_parts) == 1
        assert tool_parts[0]["name"] == "lookup"


# ---------------------------------------------------------------------------
# Output messages
# ---------------------------------------------------------------------------

class TestSetOutputMessages:
    def test_text_response(self, mock_span):
        choices = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ]
        _set_output_messages(mock_span, choices)
        result = _get_output_messages(mock_span)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["finish_reason"] == "stop"
        assert result[0]["parts"][0]["content"] == "Hello!"
        assert result[0]["parts"][0]["type"] == "text"

    def test_tool_call_response(self, mock_span):
        choices = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "NYC"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
        _set_output_messages(mock_span, choices)
        result = _get_output_messages(mock_span)

        assert len(result) == 1
        msg = result[0]
        assert msg["finish_reason"] == "tool_call"
        tool_parts = [p for p in msg["parts"] if p["type"] == "tool_call"]
        assert len(tool_parts) == 1
        assert tool_parts[0]["name"] == "get_weather"
        assert tool_parts[0]["id"] == "call_abc"

    def test_none_message_in_choice(self, mock_span):
        """Choice with message=None should produce empty output array."""
        choices = [
            {"index": 0, "message": None, "finish_reason": "content_filter"},
        ]
        _set_output_messages(mock_span, choices)
        result = _get_output_messages(mock_span)
        assert result == []

    def test_none_choices(self, mock_span):
        """None choices should be a no-op."""
        _set_output_messages(mock_span, None)
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in mock_span._attrs

    def test_multiple_choices(self, mock_span):
        choices = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Answer A"},
                "finish_reason": "stop",
            },
            {
                "index": 1,
                "message": {"role": "assistant", "content": "Answer B"},
                "finish_reason": "stop",
            },
        ]
        _set_output_messages(mock_span, choices)
        result = _get_output_messages(mock_span)

        assert len(result) == 2
        assert result[0]["parts"][0]["content"] == "Answer A"
        assert result[1]["parts"][0]["content"] == "Answer B"


# ---------------------------------------------------------------------------
# Completion wrappers: _set_input_messages for text completions
# ---------------------------------------------------------------------------

class TestCompletionInputMessages:
    def test_list_prompt_preserves_all_items(self, mock_span):
        """Batched prompt list should preserve all items, not collapse to first."""
        prompts = ["First prompt", "Second prompt", "Third prompt"]
        _set_completion_input_messages(mock_span, prompts)
        result = _get_input_messages(mock_span)

        assert len(result) == 3
        assert result[0]["parts"][0]["content"] == "First prompt"
        assert result[1]["parts"][0]["content"] == "Second prompt"
        assert result[2]["parts"][0]["content"] == "Third prompt"

    def test_single_string_prompt(self, mock_span):
        """Single string prompt should work as a single message."""
        _set_completion_input_messages(mock_span, "Hello world")
        result = _get_input_messages(mock_span)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["content"] == "Hello world"

    def test_empty_string_prompt_is_preserved(self, mock_span):
        """Empty string prompt should still be recorded, not dropped."""
        _set_completion_input_messages(mock_span, "")
        result = _get_input_messages(mock_span)

        assert len(result) == 1
        assert result[0]["parts"][0]["content"] == ""


# ---------------------------------------------------------------------------
# _set_request_attributes: headers / user serialization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# P1: _map_finish_reason
# ---------------------------------------------------------------------------

class TestMapFinishReason:
    def test_none_returns_none(self):
        """_map_finish_reason(None) must return None, NOT fabricate 'stop'."""
        assert _map_finish_reason(None) is None

    def test_empty_string_returns_none(self):
        """_map_finish_reason('') must return None."""
        assert _map_finish_reason("") is None

    def test_tool_calls_mapped_to_tool_call(self):
        """OpenAI 'tool_calls' (plural) must map to OTel 'tool_call' (singular)."""
        assert _map_finish_reason("tool_calls") == "tool_call"

    def test_stop_passthrough(self):
        assert _map_finish_reason("stop") == "stop"

    def test_length_passthrough(self):
        assert _map_finish_reason("length") == "length"

    def test_content_filter_passthrough(self):
        assert _map_finish_reason("content_filter") == "content_filter"


# ---------------------------------------------------------------------------
# P1: Tool call arguments must be parsed objects
# ---------------------------------------------------------------------------

class TestToolCallArgumentsParsed:
    def test_input_tool_call_arguments_are_parsed(self, mock_span):
        """Tool call arguments in input messages must be parsed dicts, not raw JSON strings."""
        _set_input_messages(mock_span, [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        },
                    }
                ],
            }
        ])
        result = _get_input_messages(mock_span)
        tool_part = next(p for p in result[0]["parts"] if p["type"] == "tool_call")
        # arguments must be a parsed dict, not a raw JSON string
        assert isinstance(tool_part["arguments"], dict), (
            f"Expected parsed dict, got {type(tool_part['arguments'])}: {tool_part['arguments']}"
        )
        assert tool_part["arguments"] == {"location": "NYC"}

    def test_output_tool_call_arguments_are_parsed(self, mock_span):
        """Tool call arguments in output messages must be parsed dicts."""
        choices = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "hello"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
        _set_output_messages(mock_span, choices)
        result = _get_output_messages(mock_span)
        tool_part = next(p for p in result[0]["parts"] if p["type"] == "tool_call")
        assert isinstance(tool_part["arguments"], dict), (
            f"Expected parsed dict, got {type(tool_part['arguments'])}: {tool_part['arguments']}"
        )
        assert tool_part["arguments"] == {"query": "hello"}

    def test_invalid_json_arguments_fallback_to_string(self, mock_span):
        """Non-JSON arguments string should fall back to the raw string."""
        _set_input_messages(mock_span, [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {
                            "name": "fn",
                            "arguments": "not valid json",
                        },
                    }
                ],
            }
        ])
        result = _get_input_messages(mock_span)
        tool_part = next(p for p in result[0]["parts"] if p["type"] == "tool_call")
        assert tool_part["arguments"] == "not valid json"


# ---------------------------------------------------------------------------
# P1: Multimodal content blocks mapped to OTel part types
# ---------------------------------------------------------------------------

class TestMultimodalContentMapping:
    def test_user_image_url_mapped_to_uri_part(self, mock_span):
        """OpenAI image_url content block must map to OTel uri part."""
        _set_input_messages(mock_span, [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ])
        result = _get_input_messages(mock_span)
        parts = result[0]["parts"]

        # text block mapped
        text_parts = [p for p in parts if p.get("type") == "text"]
        assert len(text_parts) == 1
        assert text_parts[0]["content"] == "What is in this image?"

        # image_url block mapped to uri part
        uri_parts = [p for p in parts if p.get("type") == "uri"]
        assert len(uri_parts) == 1, f"Expected 1 uri part, got parts: {parts}"
        assert uri_parts[0]["uri"] == "https://example.com/img.png"
        assert uri_parts[0]["modality"] == "image"

    def test_text_list_content_mapped(self, mock_span):
        """List content with text blocks must use 'content' key not 'text' key."""
        _set_input_messages(mock_span, [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                ],
            }
        ])
        result = _get_input_messages(mock_span)
        parts = result[0]["parts"]
        assert parts[0]["type"] == "text"
        assert parts[0]["content"] == "Hello"
        assert "text" not in parts[0] or parts[0].get("type") == "text"  # "text" key only for type


# ---------------------------------------------------------------------------
# P2: _get_vendor_from_url returns OTel well-known provider name values
# ---------------------------------------------------------------------------

class TestGetVendorFromUrl:
    def test_openai_default(self):
        assert _get_vendor_from_url("") == "openai"

    def test_openai_api(self):
        assert _get_vendor_from_url("https://api.openai.com/v1/") == "openai"

    def test_azure_openai(self):
        result = _get_vendor_from_url("https://myendpoint.openai.azure.com/")
        # OTel GenAiSystemValues.AZURE_AI_OPENAI — "az.ai.openai" is deprecated
        assert result == "azure.ai.openai", f"Expected 'azure.ai.openai', got '{result}'"

    def test_aws_bedrock(self):
        result = _get_vendor_from_url("https://bedrock-runtime.us-east-1.amazonaws.com/")
        assert result == "aws.bedrock", f"Expected 'aws.bedrock', got '{result}'"

    def test_google_vertex(self):
        result = _get_vendor_from_url("https://us-central1-aiplatform.googleapis.com/")
        assert result == "gcp.vertex_ai", f"Expected 'gcp.vertex_ai', got '{result}'"

    def test_openrouter(self):
        result = _get_vendor_from_url("https://openrouter.ai/api/v1/")
        assert result == "openrouter", f"Expected 'openrouter', got '{result}'"


# ---------------------------------------------------------------------------
# P2: metric_shared_attributes uses constant, not hardcoded string
# ---------------------------------------------------------------------------

class TestMetricSharedAttributes:
    def test_operation_name_key_is_constant(self):
        """gen_ai.operation.name key must use the upstream constant."""
        attrs = metric_shared_attributes(
            response_model="gpt-4",
            operation="chat",
            server_address="https://api.openai.com/v1/",
        )
        assert GenAIAttributes.GEN_AI_OPERATION_NAME in attrs, (
            f"Expected key '{GenAIAttributes.GEN_AI_OPERATION_NAME}' in attrs, "
            f"got keys: {list(attrs.keys())}"
        )


# ---------------------------------------------------------------------------
# P2: gen_ai.provider.name used instead of deprecated gen_ai.system
# ---------------------------------------------------------------------------

class TestProviderNameAttribute:
    def test_request_attributes_set_provider_name(self, mock_span):
        """_set_request_attributes must set gen_ai.provider.name, not gen_ai.system."""
        _set_request_attributes(mock_span, {"model": "gpt-4"})
        # The new attribute should be present
        assert GenAIAttributes.GEN_AI_PROVIDER_NAME in mock_span._attrs, (
            f"Expected '{GenAIAttributes.GEN_AI_PROVIDER_NAME}' in attrs, "
            f"got: {list(mock_span._attrs.keys())}"
        )


class TestRequestAttributesSerialization:
    def test_none_headers_not_recorded_as_string(self, mock_span):
        """kwargs with headers=None should not record literal 'None'."""
        _set_request_attributes(mock_span, {"model": "gpt-4"})
        val = mock_span._attrs.get(SpanAttributes.GEN_AI_HEADERS)
        assert val != "None", "Should not record literal string 'None'"

    def test_none_user_not_recorded(self, mock_span):
        """kwargs with user=None should not record attribute."""
        _set_request_attributes(mock_span, {"model": "gpt-4"})
        assert SpanAttributes.GEN_AI_USER not in mock_span._attrs


# ---------------------------------------------------------------------------
# P1: gen_ai.response.finish_reasons must be set as top-level span attribute
# ---------------------------------------------------------------------------

class TestFinishReasonsTopLevel:
    """gen_ai.response.finish_reasons is a Recommended top-level string[]
    span attribute per OTel GenAI semconv. It must be set by
    _set_response_attributes, independently of should_send_prompts."""

    def test_single_stop_reason(self, mock_span):
        """Single choice with finish_reason='stop' sets ['stop']."""
        response = {
            "model": "gpt-4",
            "id": "chatcmpl-123",
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hi"}},
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }
        _set_response_attributes(mock_span, response)
        assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS in mock_span._attrs, (
            f"Expected '{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}' in span attrs, "
            f"got: {list(mock_span._attrs.keys())}"
        )
        assert mock_span._attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)

    def test_tool_calls_mapped_to_tool_call(self, mock_span):
        """finish_reason='tool_calls' must be mapped to 'tool_call' (singular)."""
        response = {
            "model": "gpt-4",
            "id": "chatcmpl-456",
            "choices": [
                {"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant"}},
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }
        _set_response_attributes(mock_span, response)
        assert mock_span._attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("tool_call",)

    def test_multiple_choices(self, mock_span):
        """Multiple choices produce an array of mapped finish reasons."""
        response = {
            "model": "gpt-4",
            "id": "chatcmpl-789",
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant"}},
                {"index": 1, "finish_reason": "length", "message": {"role": "assistant"}},
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        _set_response_attributes(mock_span, response)
        assert mock_span._attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop", "length")

    def test_content_filter_reason(self, mock_span):
        """content_filter finish reason passes through unchanged."""
        response = {
            "model": "gpt-4",
            "id": "chatcmpl-cf",
            "choices": [
                {"index": 0, "finish_reason": "content_filter", "message": {"role": "assistant"}},
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }
        _set_response_attributes(mock_span, response)
        assert mock_span._attrs[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("content_filter",)

    def test_no_choices_no_finish_reasons(self, mock_span):
        """Response without choices should not set finish_reasons."""
        response = {
            "model": "gpt-4",
            "id": "chatcmpl-empty",
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
        }
        _set_response_attributes(mock_span, response)
        assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in mock_span._attrs

    def test_error_response_no_finish_reasons(self, mock_span):
        """Error response should not set finish_reasons."""
        response = {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}
        _set_response_attributes(mock_span, response)
        assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in mock_span._attrs
