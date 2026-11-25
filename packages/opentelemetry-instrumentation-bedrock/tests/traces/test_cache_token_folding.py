"""Cache-token folding into ``gen_ai.usage.input_tokens`` (#4449).

Bedrock reports its native input token count as the *non-cached* portion of the
prompt only. AWS documents the total as
``inputTokens + cacheReadInputTokens + cacheWriteInputTokens``, while the GenAI
semantic conventions require the opposite representation:
``gen_ai.usage.cache_read.input_tokens`` and
``gen_ai.usage.cache_write.input_tokens`` SHOULD be included in
``gen_ai.usage.input_tokens`` (semantic-conventions-genai aws-bedrock.md notes
23/24; note 28: "SHOULD include all types of input tokens, including cached
tokens").

Before this, the bedrock package emitted the raw non-cached count while the
anthropic package folded cache in, so the same cached Claude call reported
different ``gen_ai.usage.input_tokens`` / ``gen_ai.usage.total_tokens`` depending
on which package instrumented it.
"""

from unittest.mock import MagicMock

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

from opentelemetry.instrumentation.bedrock.span_utils import (
    _input_tokens_with_cache,
    _set_amazon_span_attributes,
    _set_anthropic_messages_span_attributes,
    converse_usage_record,
)

INPUT = GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
OUTPUT = GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
TOTAL = SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS
CACHE_READ = SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS
CACHE_WRITE = SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS


def _span():
    span = MagicMock()
    span.is_recording.return_value = True
    span._attrs = {}
    span.set_attribute = lambda name, value: span._attrs.__setitem__(name, value)
    return span


def _metric_params():
    mp = MagicMock()
    mp.vendor = "aws.bedrock"
    mp.model = "test-model"
    mp.is_stream = False
    mp.duration_histogram = None
    mp.token_histogram = None
    mp.start_time = 0
    return mp


MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


class TestConverseUsageRecord:
    """`converse_usage_record` backs Converse and ConverseStream usage."""

    def test_cache_read_and_write_are_folded_into_input_tokens(self):
        span = _span()

        converse_usage_record(
            span,
            {
                "usage": {
                    "inputTokens": 12,
                    "outputTokens": 40,
                    "cacheReadInputTokens": 1000,
                    "cacheWriteInputTokens": 25,
                }
            },
            _metric_params(),
        )

        assert span._attrs[INPUT] == 1037
        assert span._attrs[OUTPUT] == 40
        assert span._attrs[TOTAL] == 1077

    def test_subset_attributes_still_report_raw_cache_counts(self):
        span = _span()

        converse_usage_record(
            span,
            {
                "usage": {
                    "inputTokens": 12,
                    "outputTokens": 40,
                    "cacheReadInputTokens": 1000,
                    "cacheWriteInputTokens": 25,
                }
            },
            _metric_params(),
        )

        assert span._attrs[CACHE_READ] == 1000
        assert span._attrs[CACHE_WRITE] == 25

    def test_cache_write_only_is_folded(self):
        span = _span()

        converse_usage_record(
            span,
            {"usage": {"inputTokens": 4, "outputTokens": 50, "cacheWriteInputTokens": 18131}},
            _metric_params(),
        )

        assert span._attrs[INPUT] == 18135

    def test_cache_read_only_is_folded(self):
        span = _span()

        converse_usage_record(
            span,
            {"usage": {"inputTokens": 4, "outputTokens": 50, "cacheReadInputTokens": 18131}},
            _metric_params(),
        )

        assert span._attrs[INPUT] == 18135

    def test_zero_cache_counts_leave_input_tokens_unchanged(self):
        """Real Converse responses carry explicit zero cache counts."""
        span = _span()

        converse_usage_record(
            span,
            {
                "usage": {
                    "cacheReadInputTokenCount": 0,
                    "cacheReadInputTokens": 0,
                    "cacheWriteInputTokenCount": 0,
                    "cacheWriteInputTokens": 0,
                    "inputTokens": 20,
                    "outputTokens": 72,
                    "totalTokens": 92,
                }
            },
            _metric_params(),
        )

        assert span._attrs[INPUT] == 20
        assert span._attrs[TOTAL] == 92

    def test_usage_without_cache_fields_is_unchanged(self):
        span = _span()

        converse_usage_record(
            span, {"usage": {"inputTokens": 20, "outputTokens": 72}}, _metric_params()
        )

        assert span._attrs[INPUT] == 20
        assert span._attrs[TOTAL] == 92
        assert CACHE_READ not in span._attrs
        assert CACHE_WRITE not in span._attrs

    def test_missing_input_tokens_defaults_to_zero(self):
        span = _span()

        converse_usage_record(span, {"usage": {"outputTokens": 5}}, _metric_params())

        assert span._attrs[INPUT] == 0


class TestAnthropicMessagesUsage:
    """InvokeModel with the native Anthropic Messages body."""

    def test_cache_creation_and_read_are_folded_into_input_tokens(self):
        span = _span()

        _set_anthropic_messages_span_attributes(
            span,
            {"messages": MESSAGES},
            {
                "usage": {
                    "input_tokens": 4,
                    "cache_creation_input_tokens": 18131,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 50,
                }
            },
            {},
            _metric_params(),
        )

        assert span._attrs[INPUT] == 18135
        assert span._attrs[TOTAL] == 18185

    def test_usage_without_cache_fields_is_unchanged(self):
        span = _span()

        _set_anthropic_messages_span_attributes(
            span,
            {"messages": MESSAGES},
            {"usage": {"input_tokens": 16, "output_tokens": 8}},
            {},
            _metric_params(),
        )

        assert span._attrs[INPUT] == 16
        assert span._attrs[TOTAL] == 24

    def test_header_fallback_folds_cache_headers(self):
        """claude-v2 style: counts only in headers, cache counts in headers too."""
        span = _span()

        _set_anthropic_messages_span_attributes(
            span,
            {"messages": MESSAGES},
            {"completion": "hi"},
            {
                "x-amzn-bedrock-input-token-count": "4",
                "x-amzn-bedrock-output-token-count": "50",
                "x-amzn-bedrock-cache-read-input-token-count": "18131",
                "x-amzn-bedrock-cache-write-input-token-count": "0",
            },
            _metric_params(),
        )

        assert span._attrs[INPUT] == 18135


class TestAmazonStreamingUsage:
    """Nova InvokeModel stream: usage under `metadata`, cache counts in headers."""

    def test_cache_headers_are_folded_into_input_tokens(self):
        span = _span()

        _set_amazon_span_attributes(
            span,
            {"inputText": "hi"},
            {"metadata": {"usage": {"inputTokens": 30, "outputTokens": 51}}},
            {
                "x-amzn-bedrock-cache-read-input-token-count": "900",
                "x-amzn-bedrock-cache-write-input-token-count": "100",
            },
            _metric_params(),
        )

        assert span._attrs[INPUT] == 1030

    def test_without_cache_headers_input_tokens_is_unchanged(self):
        span = _span()

        _set_amazon_span_attributes(
            span,
            {"inputText": "hi"},
            {"metadata": {"usage": {"inputTokens": 30, "outputTokens": 51}}},
            {},
            _metric_params(),
        )

        assert span._attrs[INPUT] == 30

    def test_titan_results_branch_is_untouched(self):
        span = _span()

        _set_amazon_span_attributes(
            span,
            {"inputText": "hi"},
            {
                "inputText": "out",
                "inputTextTokenCount": 28,
                "results": [{"outputText": "o", "tokenCount": 5}],
            },
            {},
            _metric_params(),
        )

        assert span._attrs[INPUT] == 28


class TestInputTokensWithCacheHelper:
    @pytest.mark.parametrize(
        "input_tokens,cache_read,cache_write,expected",
        [
            (10, 0, 0, 10),
            (10, None, None, 10),
            (10, 5, None, 15),
            (10, None, 7, 17),
            (10, 5, 7, 22),
            (0, 5, 7, 12),
            (None, 5, 7, None),
        ],
    )
    def test_folds(self, input_tokens, cache_read, cache_write, expected):
        assert _input_tokens_with_cache(input_tokens, cache_read, cache_write) == expected