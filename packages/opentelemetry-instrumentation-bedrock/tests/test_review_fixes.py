"""Tests for review fix items — written BEFORE the fixes.

Each test documents the expected behaviour after the fix is applied.
Tests are expected to FAIL until the corresponding fix is implemented.

1. AI21 finish reason must not default to "unknown"
2. Redundant gen_ai.response.finish_reasons writes removed from response helpers
3. Empty [] must not be written to gen_ai.input.messages when messages key absent
4. Cohere "ERROR" must map to OTel "error" in BEDROCK_FINISH_REASON_MAP
"""

import json
from unittest.mock import MagicMock, patch

from opentelemetry.instrumentation.bedrock.span_utils import (
    BEDROCK_FINISH_REASON_MAP,
    _map_finish_reason,
    _set_span_completions_attributes,
    _set_generations_span_attributes,
    _set_anthropic_response_span_attributes,
    _set_llama_response_span_attributes,
    _set_amazon_response_span_attributes,
    _set_imported_model_response_span_attributes,
    set_converse_input_prompt_span_attributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_span():
    """Return a mock span that records set_attribute calls."""
    span = MagicMock()
    span.is_recording.return_value = True
    attrs = {}

    def set_attr(key, value):
        attrs[key] = value

    span.set_attribute.side_effect = set_attr
    span._attrs = attrs
    return span


def _get_attr(span, key):
    return span._attrs.get(key)


# ═══════════════════════════════════════════════════════════════════════════
# 1. AI21 finish reason must not default to "unknown"
# ═══════════════════════════════════════════════════════════════════════════


class TestAI21FinishReasonNoUnknown:
    """AI21 _set_span_completions_attributes must not emit 'unknown' as finish reason."""

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_missing_reason_key_yields_none_not_unknown(self, _mock):
        """When finishReason dict has no 'reason' key, finish_reasons should omit it (not emit 'unknown')."""
        span = _mock_span()
        response_body = {
            "completions": [
                {
                    "finishReason": {},  # no "reason" key
                    "data": {"text": "hello"},
                }
            ]
        }
        _set_span_completions_attributes(span, response_body)

        fr_attr = _get_attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS)
        # Should be empty tuple or not set — NOT ("unknown",)
        if fr_attr is not None:
            assert "unknown" not in fr_attr, (
                "finish_reasons must not contain 'unknown' when reason key is missing"
            )

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_present_reason_key_still_mapped_in_output_messages(self, _mock):
        """When finishReason has a 'reason' key, it should appear in output_messages."""
        span = _mock_span()
        response_body = {
            "completions": [
                {
                    "finishReason": {"reason": "endoftext"},
                    "data": {"text": "hello"},
                }
            ]
        }
        _set_span_completions_attributes(span, response_body)

        # finish_reasons is NOT written here (single-writer is _set_finish_reasons_unconditionally)
        assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in span._attrs
        # But the output_messages should include finish_reason in the message
        output_msgs = json.loads(_get_attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES))
        assert output_msgs[0]["finish_reason"] == "endoftext"

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_string_finishreason_mapped_in_output_messages(self, _mock):
        """When finishReason is a plain string (not dict), it should appear in output_messages."""
        span = _mock_span()
        response_body = {
            "completions": [
                {
                    "finishReason": "endoftext",
                    "data": {"text": "hello"},
                }
            ]
        }
        _set_span_completions_attributes(span, response_body)

        # finish_reasons is NOT written here (single-writer is _set_finish_reasons_unconditionally)
        assert GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in span._attrs
        # But the output_messages should include finish_reason in the message
        output_msgs = json.loads(_get_attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES))
        assert output_msgs[0]["finish_reason"] == "endoftext"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Redundant finish_reasons writes — response helpers must NOT write
#    GEN_AI_RESPONSE_FINISH_REASONS (let _set_finish_reasons_unconditionally
#    be the single writer via set_model_choice_span_attributes)
# ═══════════════════════════════════════════════════════════════════════════


class TestNoRedundantFinishReasonWrites:
    """Response helpers must not write GEN_AI_RESPONSE_FINISH_REASONS."""

    FINISH_REASONS_KEY = GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_cohere_generations_no_finish_reasons_write(self, _mock):
        span = _mock_span()
        response_body = {
            "generations": [
                {"text": "hi", "finish_reason": "COMPLETE"},
            ]
        }
        _set_generations_span_attributes(span, response_body)
        assert self.FINISH_REASONS_KEY not in span._attrs, (
            "_set_generations_span_attributes should not write finish_reasons"
        )

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_anthropic_response_no_finish_reasons_write(self, _mock):
        span = _mock_span()
        response_body = {
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
        }
        _set_anthropic_response_span_attributes(span, response_body)
        assert self.FINISH_REASONS_KEY not in span._attrs, (
            "_set_anthropic_response_span_attributes should not write finish_reasons"
        )

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_ai21_completions_no_finish_reasons_write(self, _mock):
        span = _mock_span()
        response_body = {
            "completions": [
                {"finishReason": {"reason": "endoftext"}, "data": {"text": "hi"}},
            ]
        }
        _set_span_completions_attributes(span, response_body)
        assert self.FINISH_REASONS_KEY not in span._attrs, (
            "_set_span_completions_attributes should not write finish_reasons"
        )

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_llama_response_no_finish_reasons_write(self, _mock):
        span = _mock_span()
        response_body = {
            "generation": "hi",
            "stop_reason": "end_turn",
        }
        _set_llama_response_span_attributes(span, response_body)
        assert self.FINISH_REASONS_KEY not in span._attrs, (
            "_set_llama_response_span_attributes should not write finish_reasons"
        )

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_amazon_results_no_finish_reasons_write(self, _mock):
        span = _mock_span()
        response_body = {
            "results": [
                {"outputText": "hi", "completionReason": "FINISH"},
            ]
        }
        _set_amazon_response_span_attributes(span, response_body)
        assert self.FINISH_REASONS_KEY not in span._attrs, (
            "_set_amazon_response_span_attributes (results) should not write finish_reasons"
        )

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_amazon_output_no_finish_reasons_write(self, _mock):
        span = _mock_span()
        response_body = {
            "output": {"message": {"content": [{"text": "hi"}], "role": "assistant"}},
            "stopReason": "end_turn",
        }
        _set_amazon_response_span_attributes(span, response_body)
        assert self.FINISH_REASONS_KEY not in span._attrs, (
            "_set_amazon_response_span_attributes (output) should not write finish_reasons"
        )

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_imported_model_no_finish_reasons_write(self, _mock):
        span = _mock_span()
        response_body = {
            "generation": "hi",
            "stop_reason": "end_turn",
        }
        _set_imported_model_response_span_attributes(span, response_body)
        assert self.FINISH_REASONS_KEY not in span._attrs, (
            "_set_imported_model_response_span_attributes should not write finish_reasons"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Empty [] must not be written to gen_ai.input.messages
# ═══════════════════════════════════════════════════════════════════════════


class TestNoEmptyInputMessages:
    """set_converse_input_prompt_span_attributes must not write [] when messages absent."""

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_no_messages_key_does_not_write_input_messages(self, _mock):
        """When kwargs has no 'messages' key, gen_ai.input.messages should not be set."""
        span = _mock_span()
        kwargs = {"modelId": "us.amazon.nova-lite-v1:0"}  # no "messages"
        set_converse_input_prompt_span_attributes(kwargs, span)

        input_msgs = _get_attr(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES)
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

        input_msgs = _get_attr(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert input_msgs is not None
        parsed = json.loads(input_msgs)
        assert len(parsed) == 1
        assert parsed[0]["role"] == "user"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Cohere "ERROR" must map to OTel "error"
# ═══════════════════════════════════════════════════════════════════════════


class TestCohereErrorMapping:
    """Cohere 'ERROR' finish reason must be mapped to OTel 'error'."""

    def test_error_in_map(self):
        assert "ERROR" in BEDROCK_FINISH_REASON_MAP, (
            "BEDROCK_FINISH_REASON_MAP must include 'ERROR'"
        )

    def test_error_maps_to_otel_error(self):
        assert _map_finish_reason("ERROR") == "error"

    def test_existing_cohere_mappings_unchanged(self):
        """Ensure existing Cohere mappings are not broken."""
        assert _map_finish_reason("COMPLETE") == "stop"
        assert _map_finish_reason("MAX_TOKENS") == "length"
        assert _map_finish_reason("TOOL_CALL") == "tool_call"
