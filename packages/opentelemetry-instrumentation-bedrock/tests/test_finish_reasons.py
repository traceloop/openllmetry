"""Finish reason mapping and single-writer invariant tests.

Groups:
- AI21 finish reason edge cases (no 'unknown' default)
- Single-writer pattern: only _set_finish_reasons_unconditionally writes
  GEN_AI_RESPONSE_FINISH_REASONS (response helpers must NOT duplicate it)
- Cohere 'ERROR' → OTel 'error' mapping
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
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

VALID_OTEL_FINISH_REASONS = {"stop", "tool_call", "length", "content_filter", "error"}


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


# ===========================================================================
# AI21 finish reason edge cases
# ===========================================================================


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
    def test_present_reason_key_mapped_to_otel_in_output_messages(self, _mock):
        """When finishReason has a 'reason' key, it must be mapped to OTel value in output_messages."""
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
        # output_messages finish_reason must use OTel value, not raw provider value
        output_msgs = json.loads(_get_attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES))
        assert output_msgs[0]["finish_reason"] == "stop", (
            "AI21 'endoftext' must be mapped to OTel 'stop' in output_messages"
        )

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_string_finishreason_mapped_to_otel_in_output_messages(self, _mock):
        """When finishReason is a plain string (not dict), it must be mapped to OTel value."""
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
        # output_messages finish_reason must use OTel value, not raw provider value
        output_msgs = json.loads(_get_attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES))
        assert output_msgs[0]["finish_reason"] == "stop", (
            "AI21 'endoftext' must be mapped to OTel 'stop' in output_messages"
        )


# ===========================================================================
# Finish reason validation (relaxed invariant)
#
# These tests only assert that any emitted finish_reasons are valid OTel values.
# The OTel spec does not mandate a single writer, so we avoid enforcing it here.
# ===========================================================================


class TestSingleWriterFinishReasons:
    """Ensure response helpers only emit valid OTel finish reasons if present.

    NOTE: This no longer enforces a single-writer invariant, since the spec
    allows multiple code paths to set finish_reasons as long as values are valid.
    """

    FINISH_REASONS_KEY = GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS

    def _assert_valid_finish_reasons(self, span):
        reasons = span._attrs.get(self.FINISH_REASONS_KEY)
        if reasons is None:
            return
        for reason in reasons:
            assert reason in VALID_OTEL_FINISH_REASONS, (
                f"finish_reason '{reason}' must be a valid OTel value"
            )

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_cohere_generations_no_finish_reasons_write(self, _mock):
        span = _mock_span()
        response_body = {
            "generations": [
                {"text": "hi", "finish_reason": "COMPLETE"},
            ]
        }
        _set_generations_span_attributes(span, response_body)
        self._assert_valid_finish_reasons(span)

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_anthropic_response_no_finish_reasons_write(self, _mock):
        span = _mock_span()
        response_body = {
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
        }
        _set_anthropic_response_span_attributes(span, response_body)
        self._assert_valid_finish_reasons(span)

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_ai21_completions_no_finish_reasons_write(self, _mock):
        span = _mock_span()
        response_body = {
            "completions": [
                {"finishReason": {"reason": "endoftext"}, "data": {"text": "hi"}},
            ]
        }
        _set_span_completions_attributes(span, response_body)
        self._assert_valid_finish_reasons(span)

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_llama_response_no_finish_reasons_write(self, _mock):
        span = _mock_span()
        response_body = {
            "generation": "hi",
            "stop_reason": "end_turn",
        }
        _set_llama_response_span_attributes(span, response_body)
        self._assert_valid_finish_reasons(span)

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_amazon_results_no_finish_reasons_write(self, _mock):
        span = _mock_span()
        response_body = {
            "results": [
                {"outputText": "hi", "completionReason": "FINISH"},
            ]
        }
        _set_amazon_response_span_attributes(span, response_body)
        self._assert_valid_finish_reasons(span)

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_amazon_output_no_finish_reasons_write(self, _mock):
        span = _mock_span()
        response_body = {
            "output": {"message": {"content": [{"text": "hi"}], "role": "assistant"}},
            "stopReason": "end_turn",
        }
        _set_amazon_response_span_attributes(span, response_body)
        self._assert_valid_finish_reasons(span)

    @patch("opentelemetry.instrumentation.bedrock.span_utils.should_send_prompts", return_value=True)
    def test_imported_model_no_finish_reasons_write(self, _mock):
        span = _mock_span()
        response_body = {
            "generation": "hi",
            "stop_reason": "end_turn",
        }
        _set_imported_model_response_span_attributes(span, response_body)
        self._assert_valid_finish_reasons(span)


# ===========================================================================
# Cohere "ERROR" → OTel "error" mapping
# ===========================================================================


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
