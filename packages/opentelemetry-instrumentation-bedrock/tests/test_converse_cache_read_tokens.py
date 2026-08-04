"""Reproduces silent loss of gen_ai.usage.cache_read.input_tokens and
gen_ai.usage.cache_creation.input_tokens on the Converse API path.

`converse_usage_record` (opentelemetry/instrumentation/bedrock/span_utils.py)
references `opentelemetry.semconv_ai.SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS`
and `SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS`, neither of which
exists on that class — the constants only live on upstream `GenAIAttributes`
(opentelemetry.semconv._incubating.attributes.gen_ai_attributes).
Callers of `converse_usage_record` are wrapped in `@dont_throw`
(see `_handle_converse` in `__init__.py`), so in production the resulting
AttributeError is swallowed and the attribute never reaches the span with no
error surfaced anywhere.

This test calls `converse_usage_record` directly (undecorated) so the
AttributeError surfaces instead of being swallowed.
"""

from unittest.mock import MagicMock

from opentelemetry.instrumentation.bedrock.span_utils import converse_usage_record
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


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


class _FakeMetricParams:
    """Minimal stand-in for bedrock.MetricParams with histograms disabled."""

    def __init__(self):
        self.vendor = "AWS"
        self.model = "anthropic.claude-3-5-haiku-20241022-v1:0"
        self.is_stream = False
        self.duration_histogram = None
        self.token_histogram = None


class TestConverseUsageRecordCacheReadTokens:
    def test_with_cache_read_tokens_sets_cache_read_attribute(self):
        span = _mock_span()
        response = {
            "usage": {
                "inputTokens": 4,
                "outputTokens": 50,
                "cacheReadInputTokens": 18131,
            }
        }
        converse_usage_record(span, response, _FakeMetricParams())
        assert (
            _get_attr(span, GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS)
            == 18131
        )


class TestConverseUsageRecordCacheWriteTokens:
    def test_with_cache_write_tokens_sets_cache_creation_attribute(self):
        span = _mock_span()
        response = {
            "usage": {
                "inputTokens": 4,
                "outputTokens": 50,
                "cacheWriteInputTokens": 2048,
            }
        }
        converse_usage_record(span, response, _FakeMetricParams())
        assert (
            _get_attr(
                span, GenAIAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS
            )
            == 2048
        )
