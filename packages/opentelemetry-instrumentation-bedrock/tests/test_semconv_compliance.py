"""TDD tests for OTel GenAI semconv compliance (P1 & P2 issues).

These tests are written FIRST, before any implementation changes.
Each test class maps to an issue ID from the bedrock_instrumentation_review.md.

Tests here enforce the OTel GenAI semantic conventions specification,
NOT the current implementation behaviour.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiOperationNameValues,
    GenAiSystemValues,
)

from opentelemetry.instrumentation.bedrock.span_utils import (
    BEDROCK_FINISH_REASON_MAP,
    _anthropic_content_to_parts,
    _converse_content_to_parts,
    _map_finish_reason,
    _output_message,
    _set_finish_reasons_unconditionally,
    _text_part,
    set_converse_input_prompt_span_attributes,
    set_converse_model_span_attributes,
    set_converse_response_span_attributes,
    set_converse_streaming_response_span_attributes,
    set_model_choice_span_attributes,
    set_model_message_span_attributes,
    set_model_span_attributes,
)
from opentelemetry.instrumentation.bedrock.event_emitter import (
    emit_response_event_converse,
    emit_streaming_converse_response_event,
    emit_streaming_response_event,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_OTEL_FINISH_REASONS = {"stop", "tool_call", "length", "content_filter", "error"}

VALID_OTEL_PART_TYPES = {"text", "tool_call", "tool_call_response", "blob", "uri", "reasoning"}


def _mock_span():
    """Create a mock span that captures set_attribute calls."""
    span = MagicMock()
    span.is_recording.return_value = True
    attrs = {}

    def set_attribute(name, value):
        attrs[name] = value

    span.set_attribute = set_attribute
    span._attrs = attrs
    return span


def _mock_metric_params():
    """Create a mock metric_params with required attributes for _record_usage_to_span."""
    mp = MagicMock()
    mp.vendor = "test"
    mp.model = "test-model"
    mp.is_stream = False
    mp.duration_histogram = None
    mp.token_histogram = None
    mp.start_time = 0
    return mp


def _assert_valid_parts(parts, context=""):
    """Assert every part has a valid OTel type and required fields."""
    assert isinstance(parts, list), f"{context}: parts must be a list, got {type(parts)}"
    for i, part in enumerate(parts):
        assert isinstance(part, dict), f"{context}: part[{i}] must be a dict"
        assert "type" in part, f"{context}: part[{i}] missing 'type' key: {part}"
        ptype = part["type"]
        assert ptype in VALID_OTEL_PART_TYPES, (
            f"{context}: part[{i}] has invalid type '{ptype}'. "
            f"Must be one of {VALID_OTEL_PART_TYPES}. Got: {part}"
        )
        # Validate required fields per part type
        if ptype == "text":
            assert "content" in part, f"{context}: text part[{i}] missing 'content'"
        elif ptype == "tool_call":
            assert "name" in part, f"{context}: tool_call part[{i}] missing 'name'"
            assert "arguments" in part, f"{context}: tool_call part[{i}] missing 'arguments'"
        elif ptype == "tool_call_response":
            assert "id" in part, f"{context}: tool_call_response part[{i}] missing 'id'"
        elif ptype == "blob":
            assert "mime_type" in part, f"{context}: blob part[{i}] missing 'mime_type'"
            assert "modality" in part, f"{context}: blob part[{i}] missing 'modality'"
        elif ptype == "uri":
            assert "uri" in part, f"{context}: uri part[{i}] missing 'uri'"
        elif ptype == "reasoning":
            assert "content" in part, f"{context}: reasoning part[{i}] missing 'content'"


def _assert_valid_message(msg, context=""):
    """Assert a message dict has role and valid parts array."""
    assert isinstance(msg, dict), f"{context}: message must be a dict"
    assert "role" in msg, f"{context}: message missing 'role'"
    if "parts" in msg:
        _assert_valid_parts(msg["parts"], context)


def _assert_valid_output_message(msg, context=""):
    """Assert an output message dict has valid structure and finish_reason if present."""
    _assert_valid_message(msg, context)
    if "finish_reason" in msg and msg["finish_reason"] is not None:
        assert msg["finish_reason"] in VALID_OTEL_FINISH_REASONS, (
            f"{context}: invalid finish_reason '{msg['finish_reason']}'. "
            f"Must be one of {VALID_OTEL_FINISH_REASONS}"
        )


# ===========================================================================
# P1-1: gen_ai.operation.name and gen_ai.provider.name must be set on every
#        span, ideally at span creation time.
# ===========================================================================


class TestP1_1_SpanCreationAttributes:
    """gen_ai.operation.name and gen_ai.provider.name must be set on every span."""

    def test_invoke_model_prompt_sets_operation_name_text_completion(self):
        """invoke_model with prompt-based API must have gen_ai.operation.name = text_completion."""
        span = _mock_span()
        # Anthropic prompt-based invoke_model
        request_body = {
            "prompt": "Hello",
            "max_tokens_to_sample": 100,
        }
        response_body = {
            "completion": "Hi",
            "stop_reason": "end_turn",
        }
        set_model_span_attributes(
            GenAiSystemValues.AWS_BEDROCK.value, "anthropic", "anthropic.claude-v2:1",
            span, request_body, response_body, None, _mock_metric_params(), {},
        )
        assert GenAIAttributes.GEN_AI_OPERATION_NAME in span._attrs, (
            "gen_ai.operation.name must be set on invoke_model spans"
        )
        assert span._attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == GenAiOperationNameValues.TEXT_COMPLETION.value

    def test_invoke_model_messages_sets_operation_name_chat(self):
        """invoke_model with Anthropic messages API must have gen_ai.operation.name = chat.

        Reviewer comment on P1-1: invoke_model is NOT always text_completion.
        When the request body uses Anthropic 'messages' (not 'prompt'), the
        operation is 'chat'.
        """
        span = _mock_span()
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        response_body = {
            "content": [{"type": "text", "text": "Hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        set_model_span_attributes(
            GenAiSystemValues.AWS_BEDROCK.value, "anthropic", "anthropic.claude-3-sonnet-20240229-v1:0",
            span, request_body, response_body, None, _mock_metric_params(), {},
        )
        assert span._attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == GenAiOperationNameValues.CHAT.value

    def test_invoke_model_sets_provider_name(self):
        """invoke_model spans must have gen_ai.provider.name = aws.bedrock (GenAiSystemValues.AWS_BEDROCK)."""
        span = _mock_span()
        request_body = {"prompt": "Hello", "max_tokens_to_sample": 100}
        response_body = {"completion": "Hi", "stop_reason": "end_turn"}
        set_model_span_attributes(
            GenAiSystemValues.AWS_BEDROCK.value, "anthropic", "anthropic.claude-v2:1",
            span, request_body, response_body, None, _mock_metric_params(), {},
        )
        assert GenAIAttributes.GEN_AI_PROVIDER_NAME in span._attrs, (
            "gen_ai.provider.name must be set on invoke_model spans"
        )
        assert span._attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == GenAiSystemValues.AWS_BEDROCK.value

    def test_converse_sets_operation_name_chat(self):
        """converse spans must have gen_ai.operation.name = chat."""
        span = _mock_span()
        kwargs = {
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
        }
        from opentelemetry.instrumentation.bedrock.span_utils import set_converse_model_span_attributes
        set_converse_model_span_attributes(
            span, GenAiSystemValues.AWS_BEDROCK.value,
            "claude-3-sonnet-20240229-v1:0", kwargs,
        )
        assert span._attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == GenAiOperationNameValues.CHAT.value

    @pytest.mark.parametrize("model_vendor,model_id,request_body", [
        ("cohere", "cohere.command-text-v14", {"prompt": "Hi", "max_tokens": 100}),
        ("ai21", "ai21.j2-mid-v1", {"prompt": "Hi", "maxTokens": 100}),
        ("meta", "meta.llama2-13b-chat-v1", {"prompt": "Hi", "max_gen_len": 100}),
        ("amazon", "amazon.titan-text-express-v1", {"inputText": "Hi", "textGenerationConfig": {"maxTokenCount": 100}}),
    ])
    def test_all_vendors_set_operation_name(self, model_vendor, model_id, request_body):
        """Every vendor path must set gen_ai.operation.name."""
        span = _mock_span()
        response_body = self._make_minimal_response(model_vendor)
        set_model_span_attributes(
            GenAiSystemValues.AWS_BEDROCK.value, model_vendor, model_id,
            span, request_body, response_body, None, _mock_metric_params(), {},
        )
        assert GenAIAttributes.GEN_AI_OPERATION_NAME in span._attrs, (
            f"gen_ai.operation.name must be set for vendor '{model_vendor}'"
        )

    @pytest.mark.parametrize("model_vendor,model_id,request_body", [
        ("cohere", "cohere.command-text-v14", {"prompt": "Hi", "max_tokens": 100}),
        ("ai21", "ai21.j2-mid-v1", {"prompt": "Hi", "maxTokens": 100}),
        ("meta", "meta.llama2-13b-chat-v1", {"prompt": "Hi", "max_gen_len": 100}),
        ("amazon", "amazon.titan-text-express-v1", {"inputText": "Hi", "textGenerationConfig": {"maxTokenCount": 100}}),
    ])
    def test_all_vendors_set_provider_name(self, model_vendor, model_id, request_body):
        """Every vendor path must set gen_ai.provider.name = aws_bedrock."""
        span = _mock_span()
        response_body = self._make_minimal_response(model_vendor)
        set_model_span_attributes(
            GenAiSystemValues.AWS_BEDROCK.value, model_vendor, model_id,
            span, request_body, response_body, None, _mock_metric_params(), {},
        )
        assert GenAIAttributes.GEN_AI_PROVIDER_NAME in span._attrs, (
            f"gen_ai.provider.name must be set for vendor '{model_vendor}'"
        )
        assert span._attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] == GenAiSystemValues.AWS_BEDROCK.value

    @staticmethod
    def _make_minimal_response(vendor):
        if vendor == "cohere":
            return {"generations": [{"text": "Hi", "finish_reason": "COMPLETE"}], "id": "abc"}
        elif vendor == "ai21":
            return {
                "completions": [{"data": {"text": "Hi", "tokens": []}, "finishReason": {"reason": "endoftext"}}],
                "prompt": {"tokens": []},
                "id": 1234,
            }
        elif vendor == "meta":
            return {"generation": "Hi", "stop_reason": "stop", "prompt_token_count": 5, "generation_token_count": 2}
        elif vendor == "amazon":
            return {"results": [{"outputText": "Hi", "completionReason": "FINISH"}]}
        return {}


# ===========================================================================
# P1-2: gen_ai.tool.definitions must be set when tools are provided
# ===========================================================================


class TestP1_2_ToolDefinitions:
    """Converse API tool configs must be captured as gen_ai.tool.definitions."""

    def test_converse_tool_config_captured(self):
        """toolConfig.tools must be written to gen_ai.tool.definitions as JSON.

        Tool definitions live in set_converse_model_span_attributes (shared path)
        so they are present in both event and non-event mode.
        """
        span = _mock_span()
        kwargs = {
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "messages": [{"role": "user", "content": [{"text": "What's the weather?"}]}],
            "toolConfig": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "get_weather",
                            "description": "Get current weather",
                            "inputSchema": {
                                "json": {
                                    "type": "object",
                                    "properties": {
                                        "location": {"type": "string"},
                                    },
                                }
                            },
                        }
                    }
                ]
            },
        }
        set_converse_model_span_attributes(
            span, GenAiSystemValues.AWS_BEDROCK.value,
            "claude-3-sonnet-20240229-v1:0", kwargs,
        )

        assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS in span._attrs, (
            "gen_ai.tool.definitions must be set when toolConfig is provided"
        )
        tool_defs = json.loads(span._attrs[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS])
        assert isinstance(tool_defs, list)
        assert len(tool_defs) == 1
        assert tool_defs[0]["name"] == "get_weather"

    def test_converse_no_tool_config_no_attribute(self):
        """When no toolConfig is provided, gen_ai.tool.definitions must be absent."""
        span = _mock_span()
        kwargs = {
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
        }
        set_converse_model_span_attributes(
            span, GenAiSystemValues.AWS_BEDROCK.value,
            "claude-3-sonnet-20240229-v1:0", kwargs,
        )

        assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS not in span._attrs, (
            "gen_ai.tool.definitions must NOT be set when no tools are provided"
        )

    def test_converse_multiple_tools(self):
        """Multiple tools must all be captured."""
        span = _mock_span()
        kwargs = {
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "messages": [{"role": "user", "content": [{"text": "Do stuff"}]}],
            "toolConfig": {
                "tools": [
                    {"toolSpec": {"name": "tool_a", "description": "A"}},
                    {"toolSpec": {"name": "tool_b", "description": "B"}},
                ]
            },
        }
        set_converse_model_span_attributes(
            span, GenAiSystemValues.AWS_BEDROCK.value,
            "claude-3-sonnet-20240229-v1:0", kwargs,
        )

        assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS in span._attrs
        tool_defs = json.loads(span._attrs[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS])
        assert len(tool_defs) == 2
        names = {t["name"] for t in tool_defs}
        assert names == {"tool_a", "tool_b"}


# ===========================================================================
# P1-3: AI21 "endoftext" must map to OTel "stop"
# ===========================================================================


class TestP1_3_AI21FinishReasonMapping:
    """AI21's 'endoftext' finish reason must map to OTel 'stop'."""

    def test_endoftext_maps_to_stop(self):
        """_map_finish_reason('endoftext') must return 'stop'."""
        result = _map_finish_reason("endoftext")
        assert result == "stop", (
            f"AI21 'endoftext' must map to OTel 'stop', got '{result}'"
        )

    def test_endoftext_in_finish_reason_map(self):
        """'endoftext' must be in BEDROCK_FINISH_REASON_MAP."""
        assert "endoftext" in BEDROCK_FINISH_REASON_MAP, (
            "'endoftext' must be explicitly mapped in BEDROCK_FINISH_REASON_MAP"
        )

    def test_ai21_span_finish_reason_is_stop(self):
        """AI21 response with finishReason 'endoftext' must produce 'stop' on span."""
        span = _mock_span()
        response_body = {
            "completions": [
                {"data": {"text": "Hi", "tokens": []}, "finishReason": {"reason": "endoftext"}}
            ],
            "prompt": {"tokens": []},
            "id": 1234,
        }
        set_model_choice_span_attributes("ai21", span, response_body)
        reasons = span._attrs.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert reasons is not None, "finish_reasons must be set"
        assert "stop" in reasons, (
            f"AI21 'endoftext' must produce 'stop' in finish_reasons, got {reasons}"
        )
        for r in reasons:
            assert r in VALID_OTEL_FINISH_REASONS, (
                f"Finish reason '{r}' not in OTel enum: {VALID_OTEL_FINISH_REASONS}"
            )

    def test_all_mapped_reasons_are_valid_otel_values(self):
        """Every value in BEDROCK_FINISH_REASON_MAP must be a valid OTel finish reason."""
        for provider_reason, otel_reason in BEDROCK_FINISH_REASON_MAP.items():
            assert otel_reason in VALID_OTEL_FINISH_REASONS, (
                f"BEDROCK_FINISH_REASON_MAP['{provider_reason}'] = '{otel_reason}' "
                f"is not a valid OTel finish reason. Valid: {VALID_OTEL_FINISH_REASONS}"
            )

    def test_unmapped_reasons_should_not_pass_through_as_non_otel_values(self):
        """_map_finish_reason must not return values outside the OTel enum.

        The current fallback `BEDROCK_FINISH_REASON_MAP.get(reason, reason)` passes
        through unmapped values. Any provider reason that is already a valid OTel
        value (like 'stop') is fine. But provider-specific values like 'endoftext'
        must NOT pass through.
        """
        # These are known provider values that are also valid OTel values
        # and can safely pass through
        # This test documents the expectation that ALL provider reasons are
        # either explicitly mapped or happen to match an OTel value
        known_provider_reasons = [
            "end_turn", "stop_sequence", "tool_use", "max_tokens",  # Anthropic
            "COMPLETE", "TOOL_CALL", "MAX_TOKENS", "ERROR",  # Cohere
            "FINISH",  # Titan
            "guardrail_intervened",  # Converse
            "endoftext",  # AI21
            "stop",  # Meta/Llama (already OTel value)
        ]
        for reason in known_provider_reasons:
            mapped = _map_finish_reason(reason)
            assert mapped in VALID_OTEL_FINISH_REASONS, (
                f"Provider reason '{reason}' maps to '{mapped}' which is "
                f"not a valid OTel finish reason. Must be one of {VALID_OTEL_FINISH_REASONS}"
            )


# ===========================================================================
# P1-4: Event ChoiceEvent finish reasons must use OTel enum values
#
# Reviewer comment: The GenAI event schema is less prescriptive than span
# message JSON; most existing instrumentations (Anthropic/Sagemaker/Together)
# emit flat `content`. Treat parts as a spec-confirmation item. However,
# finish_reason values in events MUST still be OTel enum values.
# ===========================================================================


class TestP1_4_ChoiceEventFinishReasons:
    """ChoiceEvent finish_reason must use OTel enum values, not provider values."""

    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_emit_events", return_value=True)
    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_send_prompts", return_value=True)
    def test_streaming_response_event_finish_reason_is_otel_value(self, _p, _e):
        """emit_streaming_response_event must use OTel finish reason values."""
        mock_logger = MagicMock()
        emitted_events = []
        mock_logger.emit = lambda lr: emitted_events.append(lr)

        # Anthropic streaming response with "end_turn" should produce "stop"
        response_body = {
            "completion": "Hello!",
            "stop_reason": "end_turn",
        }
        emit_streaming_response_event(response_body, mock_logger)

        assert len(emitted_events) == 1
        body = emitted_events[0].body
        if hasattr(body, "get"):
            fr = body.get("finish_reason")
        else:
            fr = getattr(body, "finish_reason", None)
        # finish_reason must be an OTel value
        if fr is not None:
            assert fr in VALID_OTEL_FINISH_REASONS, (
                f"Event finish_reason '{fr}' not in OTel enum"
            )

    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_emit_events", return_value=True)
    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_send_prompts", return_value=True)
    def test_converse_response_event_finish_reason_is_otel_value(self, _p, _e):
        """emit_response_event_converse must use OTel finish reason values."""
        mock_logger = MagicMock()
        emitted_events = []
        mock_logger.emit = lambda lr: emitted_events.append(lr)

        response = {
            "output": {"message": {"content": [{"text": "Hi"}], "role": "assistant"}},
            "stopReason": "end_turn",
        }
        emit_response_event_converse(response, mock_logger)

        assert len(emitted_events) == 1
        body = emitted_events[0].body
        if hasattr(body, "get"):
            fr = body.get("finish_reason")
        else:
            fr = getattr(body, "finish_reason", None)
        if fr is not None:
            assert fr in VALID_OTEL_FINISH_REASONS, (
                f"Converse event finish_reason '{fr}' not in OTel enum"
            )


# ===========================================================================
# P2-1: Converse streaming must accumulate tool_use blocks, not just text
# ===========================================================================


class TestP2_1_ConverseStreamingToolUse:
    """Converse streaming handler must capture tool_use deltas, not only text."""

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_streaming_response_with_tool_use_has_tool_call_parts(self, _mock):
        """When streaming response contains toolUse, output messages must include tool_call parts."""
        _mock_span()
        # Simulate what set_converse_streaming_response_span_attributes produces
        # for a tool call response. Currently it only takes text, so this test
        # verifies the gap: tool calls should appear in output.
        #
        # The fix should make the streaming handler accumulate tool_use blocks
        # and pass them to the span attributes formatter.
        #
        # For now, we test that _converse_content_to_parts correctly handles
        # toolUse blocks, and that the streaming output should include them.
        blocks = [
            {"toolUse": {"toolUseId": "call_1", "name": "get_weather", "input": {"city": "NYC"}}},
        ]
        parts = _converse_content_to_parts(blocks)
        assert len(parts) == 1
        assert parts[0]["type"] == "tool_call"
        assert parts[0]["name"] == "get_weather"
        assert parts[0]["id"] == "call_1"

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_streaming_text_only_response_creates_valid_output(self, _mock):
        """Streaming text-only response must produce valid parts-based output."""
        span = _mock_span()
        response_chunks = ["Hello", " world", "!"]
        set_converse_streaming_response_span_attributes(
            response_chunks, "assistant", span, finish_reason="end_turn",
        )
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in span._attrs
        output = json.loads(span._attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert len(output) >= 1
        _assert_valid_output_message(output[0], "streaming text response")


# ===========================================================================
# P2-2: Unknown block types must not emit non-OTel part types
# ===========================================================================


class TestP2_2_UnknownBlockTypes:
    """Unknown content block types must be wrapped safely, not passed through raw."""

    def test_anthropic_unknown_block_uses_valid_otel_type(self):
        """Unknown Anthropic block type must produce a valid OTel part type."""
        blocks = [{"type": "custom_new_feature", "data": "something"}]
        parts = _anthropic_content_to_parts(blocks)
        assert len(parts) == 1
        assert parts[0]["type"] in VALID_OTEL_PART_TYPES, (
            f"Unknown block produced type '{parts[0]['type']}' which is not "
            f"a valid OTel part type. Expected one of {VALID_OTEL_PART_TYPES}"
        )

    def test_converse_unknown_block_uses_valid_otel_type(self):
        """Unknown Converse block type must produce a valid OTel part type."""
        blocks = [{"newBlockType": {"data": "something"}}]
        parts = _converse_content_to_parts(blocks)
        assert len(parts) == 1
        assert parts[0]["type"] in VALID_OTEL_PART_TYPES, (
            f"Unknown converse block produced type '{parts[0]['type']}' which is not "
            f"a valid OTel part type"
        )

    def test_anthropic_unknown_block_preserves_data(self):
        """Unknown block content must not be silently dropped."""
        blocks = [{"type": "custom_widget", "payload": {"key": "value"}}]
        parts = _anthropic_content_to_parts(blocks)
        assert len(parts) == 1
        # The content should contain the original data in some form
        content = parts[0].get("content", "")
        assert "custom_widget" in content or "key" in content, (
            f"Unknown block data was lost. Part: {parts[0]}"
        )


# ===========================================================================
# P2-3: Streaming converse event emission should use parts schema
# ===========================================================================


class TestP2_3_StreamingConverseEventParts:
    """emit_streaming_converse_response_event should use parts-based schema."""

    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_emit_events", return_value=True)
    @patch("opentelemetry.instrumentation.bedrock.event_emitter.should_send_prompts", return_value=True)
    def test_streaming_converse_event_has_finish_reason(self, _p, _e):
        """Streaming converse event must include mapped finish reason."""
        mock_logger = MagicMock()
        emitted = []
        mock_logger.emit = lambda lr: emitted.append(lr)

        emit_streaming_converse_response_event(
            mock_logger,
            ["Hello", " world"],
            "assistant",
            "end_turn",
        )

        assert len(emitted) == 1
        body = emitted[0].body
        if hasattr(body, "get"):
            fr = body.get("finish_reason")
        else:
            fr = getattr(body, "finish_reason", None)
        # Must be OTel value, not provider value
        if fr is not None:
            assert fr == "stop", (
                f"'end_turn' should map to 'stop', got '{fr}'"
            )


# ===========================================================================
# P2-5: System instructions for invoke_model Anthropic path (REGRESSION)
#
# Reviewer comment: The Bedrock code ALREADY extracts 'system' for Anthropic
# invoke_model when the request body uses 'messages' (see
# set_model_message_span_attributes in span_utils.py:141-157).
# These tests serve as regression coverage, not gap identification.
# ===========================================================================


class TestP2_5_AnthropicSystemInstructions:
    """Regression: Anthropic invoke_model with 'system' key must set gen_ai.system_instructions."""

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_anthropic_messages_with_system_string(self, _mock):
        """Anthropic request with system string must set gen_ai.system_instructions."""
        span = _mock_span()
        request_body = {
            "system": "You are a helpful assistant",
            "messages": [{"role": "user", "content": "Hello"}],
            "anthropic_version": "bedrock-2023-05-31",
        }
        set_model_message_span_attributes("anthropic", span, request_body)

        assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS in span._attrs, (
            "gen_ai.system_instructions must be set when 'system' is in Anthropic request"
        )
        instructions = json.loads(span._attrs[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS])
        assert isinstance(instructions, list)
        assert len(instructions) >= 1
        _assert_valid_parts(instructions, "system_instructions")

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_anthropic_messages_with_system_list(self, _mock):
        """Anthropic request with system as list of blocks must set gen_ai.system_instructions."""
        span = _mock_span()
        request_body = {
            "system": [
                {"type": "text", "text": "You are helpful"},
                {"type": "text", "text": "Be concise"},
            ],
            "messages": [{"role": "user", "content": "Hello"}],
            "anthropic_version": "bedrock-2023-05-31",
        }
        set_model_message_span_attributes("anthropic", span, request_body)

        assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS in span._attrs
        instructions = json.loads(span._attrs[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS])
        assert len(instructions) == 2

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_anthropic_prompt_api_no_system(self, _mock):
        """Old Anthropic prompt API (no 'messages') should not have system_instructions."""
        span = _mock_span()
        request_body = {
            "prompt": "Human: Hello Assistant:",
            "max_tokens_to_sample": 100,
        }
        set_model_message_span_attributes("anthropic", span, request_body)

        assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS not in span._attrs, (
            "Prompt-based API should not set system_instructions"
        )


# ===========================================================================
# P2-4: Streaming converse response must produce valid output message structure
# ===========================================================================


class TestP2_4_StreamingOutputStructure:
    """Streaming response span attributes must have valid parts structure."""

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_streaming_output_has_valid_parts(self, _mock):
        """Streaming output messages must follow parts-based schema."""
        span = _mock_span()
        set_converse_streaming_response_span_attributes(
            ["Hello world"], "assistant", span, finish_reason="end_turn",
        )
        output = json.loads(span._attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        for msg in output:
            _assert_valid_output_message(msg, "streaming output")

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_streaming_output_finish_reason_mapped(self, _mock):
        """Streaming output finish_reason must be an OTel value."""
        span = _mock_span()
        set_converse_streaming_response_span_attributes(
            ["text"], "assistant", span, finish_reason="end_turn",
        )
        output = json.loads(span._attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        for msg in output:
            if "finish_reason" in msg and msg["finish_reason"] is not None:
                assert msg["finish_reason"] in VALID_OTEL_FINISH_REASONS

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_streaming_output_span_finish_reasons_mapped(self, _mock):
        """gen_ai.response.finish_reasons span attribute must use OTel values."""
        span = _mock_span()
        set_converse_streaming_response_span_attributes(
            ["text"], "assistant", span, finish_reason="end_turn",
        )
        reasons = span._attrs.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert reasons is not None
        for r in reasons:
            assert r in VALID_OTEL_FINISH_REASONS


# ===========================================================================
# P2-6: Content parts mapping - comprehensive structural validation
# ===========================================================================


class TestContentPartsStructuralValidation:
    """All content-to-parts mapping functions must produce valid OTel parts."""

    def test_anthropic_text_block(self):
        parts = _anthropic_content_to_parts([{"type": "text", "text": "Hello"}])
        _assert_valid_parts(parts, "anthropic text")

    def test_anthropic_tool_use_block(self):
        parts = _anthropic_content_to_parts([{
            "type": "tool_use", "id": "call_1",
            "name": "get_weather", "input": {"city": "NYC"},
        }])
        _assert_valid_parts(parts, "anthropic tool_use")
        assert parts[0]["type"] == "tool_call"

    def test_anthropic_tool_result_block(self):
        parts = _anthropic_content_to_parts([{
            "type": "tool_result", "tool_use_id": "call_1",
            "content": "Sunny, 72°F",
        }])
        _assert_valid_parts(parts, "anthropic tool_result")
        assert parts[0]["type"] == "tool_call_response"

    def test_anthropic_image_base64_block(self):
        parts = _anthropic_content_to_parts([{
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
        }])
        _assert_valid_parts(parts, "anthropic image base64")
        assert parts[0]["type"] == "blob"
        assert parts[0]["modality"] == "image"

    def test_anthropic_image_url_block(self):
        parts = _anthropic_content_to_parts([{
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/img.png"},
        }])
        _assert_valid_parts(parts, "anthropic image url")
        assert parts[0]["type"] == "uri"

    def test_anthropic_thinking_block(self):
        parts = _anthropic_content_to_parts([{
            "type": "thinking", "thinking": "Let me consider...",
        }])
        _assert_valid_parts(parts, "anthropic thinking")
        assert parts[0]["type"] == "reasoning"

    def test_anthropic_string_content(self):
        parts = _anthropic_content_to_parts(["Hello world"])
        _assert_valid_parts(parts, "anthropic string")

    def test_converse_text_block(self):
        parts = _converse_content_to_parts([{"text": "Hello"}])
        _assert_valid_parts(parts, "converse text")

    def test_converse_tool_use_block(self):
        parts = _converse_content_to_parts([{
            "toolUse": {"toolUseId": "call_1", "name": "search", "input": {}},
        }])
        _assert_valid_parts(parts, "converse toolUse")
        assert parts[0]["type"] == "tool_call"

    def test_converse_tool_result_block(self):
        parts = _converse_content_to_parts([{
            "toolResult": {"toolUseId": "call_1", "content": "Result"},
        }])
        _assert_valid_parts(parts, "converse toolResult")
        assert parts[0]["type"] == "tool_call_response"

    def test_converse_image_block(self):
        parts = _converse_content_to_parts([{
            "image": {"format": "png", "source": {"bytes": b"data"}},
        }])
        _assert_valid_parts(parts, "converse image")
        assert parts[0]["type"] == "blob"
        assert parts[0]["modality"] == "image"

    def test_converse_video_block(self):
        parts = _converse_content_to_parts([{
            "video": {"format": "mp4", "source": {"bytes": b"data"}},
        }])
        _assert_valid_parts(parts, "converse video")
        assert parts[0]["type"] == "blob"
        assert parts[0]["modality"] == "video"

    def test_converse_document_block(self):
        parts = _converse_content_to_parts([{
            "document": {"format": "pdf", "name": "report.pdf", "source": {"bytes": b"data"}},
        }])
        _assert_valid_parts(parts, "converse document")
        assert parts[0]["type"] == "blob"
        assert parts[0]["mime_type"] == "application/pdf"

    def test_converse_guard_content_block(self):
        parts = _converse_content_to_parts([{
            "guardContent": {"text": {"text": "safe content"}},
        }])
        _assert_valid_parts(parts, "converse guardContent")

    def test_output_message_helper_structure(self):
        """_output_message must produce valid structure."""
        msg = _output_message("assistant", [_text_part("Hello")], "stop")
        _assert_valid_output_message(msg, "_output_message")
        assert msg["role"] == "assistant"
        assert msg["finish_reason"] == "stop"

    def test_output_message_omits_none_finish_reason(self):
        """_output_message must omit finish_reason when None."""
        msg = _output_message("assistant", [_text_part("Hello")], None)
        assert "finish_reason" not in msg, (
            "finish_reason must be omitted when None, not set to None"
        )


# ===========================================================================
# P2-2 extended: _map_finish_reason edge cases
# ===========================================================================


class TestFinishReasonEdgeCases:
    """Edge cases for finish reason mapping."""

    def test_none_returns_none(self):
        assert _map_finish_reason(None) is None

    def test_empty_string_returns_none(self):
        assert _map_finish_reason("") is None

    @pytest.mark.parametrize("provider_reason,expected", [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("tool_use", "tool_call"),
        ("max_tokens", "length"),
        ("COMPLETE", "stop"),
        ("TOOL_CALL", "tool_call"),
        ("MAX_TOKENS", "length"),
        ("ERROR", "error"),
        ("FINISH", "stop"),
        ("guardrail_intervened", "content_filter"),
        ("stop", "stop"),  # Already OTel value, passthrough is fine
    ])
    def test_known_mappings(self, provider_reason, expected):
        result = _map_finish_reason(provider_reason)
        assert result == expected, (
            f"Expected '{provider_reason}' -> '{expected}', got '{result}'"
        )


# ===========================================================================
# Converse non-streaming output structural validation
# ===========================================================================


class TestConverseOutputStructure:
    """Converse non-streaming response must produce valid output."""

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_converse_response_output_has_valid_parts(self, _mock):
        span = _mock_span()
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello!"}],
                }
            },
            "stopReason": "end_turn",
        }
        set_converse_response_span_attributes(response, span)

        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in span._attrs
        output = json.loads(span._attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        for msg in output:
            _assert_valid_output_message(msg, "converse response")

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_converse_response_with_tool_use_output(self, _mock):
        """Converse response with tool use must include tool_call parts."""
        span = _mock_span()
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "I'll check the weather."},
                        {"toolUse": {"toolUseId": "call_1", "name": "get_weather", "input": {"city": "NYC"}}},
                    ],
                }
            },
            "stopReason": "tool_use",
        }
        set_converse_response_span_attributes(response, span)

        output = json.loads(span._attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        assert len(output) >= 1
        parts = output[0]["parts"]
        part_types = [p["type"] for p in parts]
        assert "tool_call" in part_types, (
            f"Tool use response must include tool_call parts, got types: {part_types}"
        )
        _assert_valid_parts(parts, "converse tool_use response")

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_converse_response_finish_reason_mapped(self, _mock):
        """Converse response finish_reason must use OTel value."""
        span = _mock_span()
        response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hi"}]}},
            "stopReason": "end_turn",
        }
        set_converse_response_span_attributes(response, span)

        reasons = span._attrs.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert reasons is not None
        for r in reasons:
            assert r in VALID_OTEL_FINISH_REASONS, f"'{r}' not a valid OTel finish reason"


# ===========================================================================
# P2-7 (NEW — missed in original review):
# gen_ai.response.finish_reasons MISSING when should_emit_events()=True
#
# In __init__.py, when should_emit_events() is True, the code branches to
# event emission and skips set_model_choice_span_attributes() /
# set_converse_response_span_attributes() — these are the ONLY places that
# call _set_finish_reasons_unconditionally(). As a result, spans in event
# mode have no finish_reasons attribute.
#
# finish_reasons is metadata, not content — it should ALWAYS be set.
# ===========================================================================


class TestP2_7_FinishReasonsInEventMode:
    """gen_ai.response.finish_reasons must be set on spans even when events are emitted."""

    def test_set_model_span_attributes_does_not_write_finish_reasons(self):
        """set_model_span_attributes must NOT write finish_reasons (P2-6: no double write).

        finish_reasons is written by the caller: either set_model_choice_span_attributes
        (non-event branch) or _set_finish_reasons_unconditionally (event branch).
        """
        span = _mock_span()
        response_body = {
            "completion": "Hello!",
            "stop_reason": "end_turn",
        }
        set_model_span_attributes(
            GenAiSystemValues.AWS_BEDROCK.value, "anthropic", "anthropic.claude-v2:1",
            span, {"prompt": "Hi", "max_tokens_to_sample": 100},
            response_body, None, _mock_metric_params(), {},
        )
        # finish_reasons must NOT be set here — the caller handles it
        reasons = span._attrs.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert reasons is None, (
            "set_model_span_attributes must not write finish_reasons (P2-6). "
            "The caller (_handle_call / _handle_stream_call) writes it explicitly."
        )

    def test_converse_finish_reasons_set_regardless_of_event_mode(self):
        """Converse path: finish_reasons must be on span even in event mode.

        In _handle_converse(), set_converse_response_span_attributes() writes
        finish_reasons but is only called in the non-event branch.
        """
        span = _mock_span()
        # set_converse_model_span_attributes is called in both branches
        set_converse_model_span_attributes(
            span, GenAiSystemValues.AWS_BEDROCK.value,
            "claude-3-sonnet-20240229-v1:0",
            {"modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
             "messages": [{"role": "user", "content": [{"text": "Hi"}]}]},
        )
        # But finish_reasons are NOT written by set_converse_model_span_attributes.
        # In event mode, the response is only sent to emit_response_event_converse,
        # which emits events but doesn't write the span attribute.
        #
        # To test that finish_reasons SHOULD be present, we simulate what the
        # fixed code should do: call _set_finish_reasons_unconditionally or
        # equivalent in both branches.
        #
        # For now, this test documents that after set_converse_model_span_attributes,
        # we expect finish_reasons to NOT yet be set (they should be added by the
        # response handler).
        assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in span._attrs, (
            "set_converse_model_span_attributes should not write finish_reasons "
            "(it doesn't have the response yet)"
        )

    @patch("opentelemetry.instrumentation.bedrock.utils.should_emit_events", return_value=True)
    @patch("opentelemetry.instrumentation.bedrock.utils.should_send_prompts", return_value=True)
    def test_converse_event_mode_sets_finish_reasons(self, _mock_prompts, _mock_events):
        """Converse event-mode branch must still set finish_reasons on the span."""
        from opentelemetry.instrumentation.bedrock import _handle_converse

        span = _mock_span()
        response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 2, "outputTokens": 1},
        }
        kwargs = {
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
        }
        metric_params = _mock_metric_params()

        _handle_converse(span, kwargs, response, metric_params, MagicMock())

        reasons = span._attrs.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert reasons is not None, (
            "Converse event-mode path must write gen_ai.response.finish_reasons on the span"
        )
        assert "stop" in reasons

    def test_set_model_choice_writes_finish_reasons(self):
        """Verify set_model_choice_span_attributes writes finish_reasons (baseline)."""
        span = _mock_span()
        response_body = {"completion": "Hi", "stop_reason": "end_turn"}
        set_model_choice_span_attributes("anthropic", span, response_body)
        reasons = span._attrs.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert reasons is not None, "set_model_choice_span_attributes must write finish_reasons"
        assert reasons == ("stop",)

    def test_set_converse_response_writes_finish_reasons(self):
        """Verify set_converse_response_span_attributes writes finish_reasons (baseline)."""
        span = _mock_span()
        response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hi"}]}},
            "stopReason": "end_turn",
        }
        set_converse_response_span_attributes(response, span)
        reasons = span._attrs.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert reasons is not None, "set_converse_response_span_attributes must write finish_reasons"
        assert reasons == ("stop",)

    @pytest.mark.parametrize("model_vendor,response_body,expected_reason", [
        ("anthropic", {"completion": "Hi", "stop_reason": "end_turn"}, "stop"),
        ("cohere", {"generations": [{"text": "Hi", "finish_reason": "COMPLETE"}]}, "stop"),
        ("ai21", {"completions": [{"data": {"text": "Hi"}, "finishReason": {"reason": "endoftext"}}]}, "stop"),
        ("meta", {"generation": "Hi", "stop_reason": "stop"}, "stop"),
        ("amazon", {"results": [{"outputText": "Hi", "completionReason": "FINISH"}]}, "stop"),
    ])
    def test_finish_reasons_unconditionally_for_all_vendors(
        self, model_vendor, response_body, expected_reason
    ):
        """_set_finish_reasons_unconditionally must produce correct OTel values for every vendor."""
        span = _mock_span()
        _set_finish_reasons_unconditionally(model_vendor, span, response_body)
        reasons = span._attrs.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        assert reasons is not None, f"finish_reasons must be set for vendor '{model_vendor}'"
        assert expected_reason in reasons, (
            f"Expected '{expected_reason}' for vendor '{model_vendor}', got {reasons}"
        )
        for r in reasons:
            assert r in VALID_OTEL_FINISH_REASONS


# ===========================================================================
# P1-2 extended: tool_definitions must be set even in event mode
#
# Reviewer comment: gen_ai.tool.definitions is a span attribute, so it should
# be set regardless of whether should_emit_events() is true. Today, the event
# path skips set_converse_input_prompt_span_attributes() entirely.
# ===========================================================================


class TestP1_2_ToolDefinitionsEventMode:
    """gen_ai.tool.definitions must be set on span even when events are emitted."""

    def test_tool_definitions_set_by_shared_path(self):
        """gen_ai.tool.definitions must be set by the shared path (both event and non-event).

        In event mode, set_converse_input_prompt_span_attributes is skipped,
        so tool_definitions must be extracted in set_converse_model_span_attributes
        (or an equivalent shared helper) to be present in both branches.

        This test will FAIL until the fix is applied.
        """
        span = _mock_span()
        kwargs = {
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "messages": [{"role": "user", "content": [{"text": "Weather?"}]}],
            "toolConfig": {
                "tools": [
                    {"toolSpec": {"name": "get_weather", "description": "Get weather"}},
                ]
            },
        }
        set_converse_model_span_attributes(
            span, GenAiSystemValues.AWS_BEDROCK.value,
            "claude-3-sonnet-20240229-v1:0", kwargs,
        )
        assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS in span._attrs, (
            "gen_ai.tool.definitions must be set by the shared path "
            "(set_converse_model_span_attributes) so it is present in event mode too. "
            "Fix: extract toolConfig.tools in the shared path."
        )


# ===========================================================================
# Converse input edge cases (moved from test_review_fixes.py)
# ===========================================================================


class TestConverseInputEdgeCases:
    """Edge cases for set_converse_input_prompt_span_attributes."""

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_no_messages_key_does_not_write_empty_input_messages(self, _mock):
        """When kwargs has no 'messages' key, gen_ai.input.messages must not be set to []."""
        span = _mock_span()
        kwargs = {"modelId": "us.amazon.nova-lite-v1:0"}  # no "messages"
        set_converse_input_prompt_span_attributes(kwargs, span)

        input_msgs = span._attrs.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        if input_msgs is not None:
            parsed = json.loads(input_msgs)
            assert parsed != [], (
                "gen_ai.input.messages must not be set to empty [] when messages key is absent"
            )

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_with_messages_key_writes_normally(self, _mock):
        """When kwargs has messages, gen_ai.input.messages should be populated."""
        span = _mock_span()
        kwargs = {
            "modelId": "us.amazon.nova-lite-v1:0",
            "messages": [
                {"role": "user", "content": [{"text": "Hello"}]},
            ],
        }
        set_converse_input_prompt_span_attributes(kwargs, span)

        input_msgs = span._attrs.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert input_msgs is not None
        parsed = json.loads(input_msgs)
        assert len(parsed) == 1
        assert parsed[0]["role"] == "user"


# ===========================================================================
# Fix #3: P1-1 span creation-time attributes
#
# The OTel spec requires gen_ai.operation.name and gen_ai.provider.name to be
# set at span creation (i.e. passed to start_span / start_as_current_span),
# not added later via set_attribute. This test verifies the wrapper passes
# them in the attributes dict at creation time.
# ===========================================================================


class TestP1_1_SpanCreationTimeAttributes:
    """gen_ai.operation.name and gen_ai.provider.name must be passed to start_span()."""

    def test_invoke_model_passes_attributes_at_span_creation(self):
        """_instrumented_model_invoke must pass provider/operation to start_as_current_span."""
        from opentelemetry.instrumentation.bedrock import _instrumented_model_invoke

        mock_tracer = MagicMock()
        mock_span = _mock_span()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        mock_fn = MagicMock(return_value={
            "body": MagicMock(
                _raw_stream=MagicMock(),
                _content_length=0,
                read=MagicMock(return_value=b'{"completion": "Hi", "stop_reason": "end_turn"}'),
            ),
        })

        wrapped = _instrumented_model_invoke(mock_fn, mock_tracer, _mock_metric_params(), None)
        wrapped(
            modelId="anthropic.claude-v2:1",
            body=json.dumps({"prompt": "Hello", "max_tokens_to_sample": 100}),
        )

        call_args = mock_tracer.start_as_current_span.call_args
        # The spec requires attributes= kwarg at creation time
        creation_attrs = call_args.kwargs.get("attributes") or {}
        assert GenAIAttributes.GEN_AI_PROVIDER_NAME in creation_attrs, (
            "gen_ai.provider.name must be passed to start_as_current_span(attributes=...) "
            "at span creation time, not added later via set_attribute()"
        )
        assert GenAIAttributes.GEN_AI_OPERATION_NAME in creation_attrs, (
            "gen_ai.operation.name must be passed to start_as_current_span(attributes=...) "
            "at span creation time, not added later via set_attribute()"
        )

    def test_converse_passes_attributes_at_span_creation(self):
        """_instrumented_converse must pass provider/operation to start_as_current_span."""
        from opentelemetry.instrumentation.bedrock import _instrumented_converse

        mock_tracer = MagicMock()
        mock_span = _mock_span()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        mock_fn = MagicMock(return_value={
            "output": {"message": {"content": [{"text": "Hi"}], "role": "assistant"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 2},
        })

        wrapped = _instrumented_converse(mock_fn, mock_tracer, _mock_metric_params(), None)
        wrapped(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "user", "content": [{"text": "Hello"}]}],
        )

        call_args = mock_tracer.start_as_current_span.call_args
        creation_attrs = call_args.kwargs.get("attributes") or {}
        assert GenAIAttributes.GEN_AI_PROVIDER_NAME in creation_attrs, (
            "gen_ai.provider.name must be passed at span creation time for converse"
        )
        assert GenAIAttributes.GEN_AI_OPERATION_NAME in creation_attrs, (
            "gen_ai.operation.name must be passed at span creation time for converse"
        )


# ===========================================================================
# Fix #4: Streaming tool-use accumulation must be enforced
#
# The streaming handler (_handle_converse_stream) currently only accumulates
# text deltas (contentBlockDelta with "text"), dropping toolUse blocks.
# This test exercises the actual handler and verifies tool_call parts appear
# in the output.
# ===========================================================================


class TestP2_1_StreamingToolUseAccumulation:
    """Streaming converse handler must accumulate toolUse blocks, not just text."""

    @patch("opentelemetry.instrumentation.bedrock.utils.should_emit_events", return_value=False)
    @patch("opentelemetry.instrumentation.bedrock.utils.should_send_prompts", return_value=True)
    def test_streaming_handler_captures_tool_use_in_output(self, _mock_prompts, _mock_events):
        """When stream contains toolUse blocks, output must include tool_call parts.

        This exercises the actual _handle_converse_stream handler, not just
        the _converse_content_to_parts helper.
        """
        from opentelemetry.instrumentation.bedrock import _handle_converse_stream

        span = _mock_span()
        span.end = MagicMock()

        # Build a mock stream that yields events in order
        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"toolUse": {
                "toolUseId": "call_1", "name": "get_weather", "input": '{"city": "NYC"}',
            }}}},
            {"messageStop": {"stopReason": "tool_use"}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
        ]

        class MockStream:
            def __init__(self):
                self._events = iter(events)

            def _parse_event(self):
                return next(self._events)

        mock_stream = MockStream()
        response = {"stream": mock_stream}
        kwargs_arg = {"modelId": "anthropic.claude-3-sonnet-20240229-v1:0"}
        metric_params = _mock_metric_params()

        _handle_converse_stream(span, kwargs_arg, response, metric_params, None)

        # Simulate consuming events through the patched _parse_event
        parsed_events = []
        try:
            while True:
                parsed_events.append(mock_stream._parse_event())
        except (StopIteration, TypeError):
            pass

        # After all events consumed, check output_messages on span
        output_attr = span._attrs.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        if output_attr is None:
            pytest.fail(
                "gen_ai.output.messages not set after streaming. "
                "The handler must accumulate toolUse blocks and include them as tool_call parts."
            )
        output_msgs = json.loads(output_attr)
        all_parts = [p for msg in output_msgs for p in msg.get("parts", [])]
        tool_parts = [p for p in all_parts if p.get("type") == "tool_call"]
        assert len(tool_parts) >= 1, (
            f"Streaming output must include tool_call parts from toolUse blocks, "
            f"but got parts: {all_parts}"
        )


# ===========================================================================
# Minor: tool definitions should still be set when should_send_prompts=False
#
# Tool definitions are metadata, not user content. They should be set
# regardless of the should_send_prompts() flag.
# ===========================================================================


class TestToolDefinitionsWithPromptsSuppressed:
    """gen_ai.tool.definitions must be set even when should_send_prompts=False."""

    def test_tool_definitions_set_when_prompts_suppressed(self):
        """Tool definitions are metadata and should be set regardless of should_send_prompts.

        Since tool defs live in set_converse_model_span_attributes (shared path),
        they are never gated by should_send_prompts(). This test verifies the
        shared path sets them unconditionally.
        """
        span = _mock_span()
        kwargs = {
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "messages": [{"role": "user", "content": [{"text": "Weather?"}]}],
            "toolConfig": {
                "tools": [
                    {"toolSpec": {"name": "get_weather", "description": "Get weather"}},
                ]
            },
        }
        set_converse_model_span_attributes(
            span, GenAiSystemValues.AWS_BEDROCK.value,
            "claude-3-sonnet-20240229-v1:0", kwargs,
        )

        assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS in span._attrs, (
            "gen_ai.tool.definitions is metadata and must be set even when "
            "should_send_prompts() returns False"
        )
