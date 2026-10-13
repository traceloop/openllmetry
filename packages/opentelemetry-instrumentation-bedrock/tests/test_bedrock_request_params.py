"""Unit tests for request-parameter span attributes on the Bedrock paths.

Regression coverage for: top_k and stop_sequences were silently dropped from
spans even though the caller sent them (converse: inferenceConfig.stopSequences
and additionalModelRequestFields.top_k; invoke_model: Anthropic Messages body
top_k / stop_sequences).
"""
from unittest.mock import MagicMock

from opentelemetry.instrumentation.bedrock.span_utils import (
    _set_anthropic_messages_span_attributes,
    set_converse_model_span_attributes,
)


def _attrs(span):
    return {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}


class TestConverseRequestParams:
    def test_top_k_and_stop_sequences_recorded(self):
        span = MagicMock()
        set_converse_model_span_attributes(
            span,
            provider="aws.bedrock",
            model="anthropic.claude-3-haiku-20240307-v1:0",
            kwargs={
                "inferenceConfig": {
                    "maxTokens": 512,
                    "temperature": 0.7,
                    "topP": 0.9,
                    "stopSequences": ["</answer>", "\n\nHuman:"],
                },
                "additionalModelRequestFields": {"top_k": 250},
            },
        )
        attrs = _attrs(span)
        assert attrs["gen_ai.request.top_k"] == 250
        assert list(attrs["gen_ai.request.stop_sequences"]) == ["</answer>", "\n\nHuman:"]
        # existing params untouched
        assert attrs["gen_ai.request.max_tokens"] == 512
        assert attrs["gen_ai.request.temperature"] == 0.7
        assert attrs["gen_ai.request.top_p"] == 0.9

    def test_absent_params_set_no_attributes(self):
        span = MagicMock()
        set_converse_model_span_attributes(
            span, provider="aws.bedrock", model="m", kwargs={"inferenceConfig": {}}
        )
        attrs = _attrs(span)
        assert "gen_ai.request.top_k" not in attrs
        assert "gen_ai.request.stop_sequences" not in attrs


class TestInvokeModelRequestParams:
    def test_top_k_and_stop_sequences_recorded(self):
        span = MagicMock()
        _set_anthropic_messages_span_attributes(
            span,
            request_body={
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 100,
                "stop_sequences": ["</answer>"],
                "messages": [{"role": "user", "content": "hi"}],
            },
            response_body={},
            headers={},
            metric_params=MagicMock(),
        )
        attrs = _attrs(span)
        assert attrs["gen_ai.request.top_k"] == 100
        assert list(attrs["gen_ai.request.stop_sequences"]) == ["</answer>"]

    def test_absent_params_set_no_attributes(self):
        span = MagicMock()
        _set_anthropic_messages_span_attributes(
            span,
            request_body={"messages": []},
            response_body={},
            headers={},
            metric_params=MagicMock(),
        )
        attrs = _attrs(span)
        assert "gen_ai.request.top_k" not in attrs
        assert "gen_ai.request.stop_sequences" not in attrs
