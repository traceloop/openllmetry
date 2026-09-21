"""Unit tests for guardrail activation detection (issue #4471).

is_guardrail_activated used to return True whenever the
``amazon-bedrock-guardrailAction`` key was absent, because
``None != "NONE"`` is True. That made every ordinary Bedrock response — from
users with no guardrail configured — look like a guardrail activation, and
``gen_ai.bedrock.guardrail.activation`` was incremented on every call.

These tests exercise the pure detection helper directly (no AWS, no cassette)
and assert that the activation metric is not incremented on a plain response.
"""

from unittest.mock import MagicMock

import pytest

from opentelemetry.instrumentation.bedrock.guardrail import (
    guardrail_converse,
    is_guardrail_activated,
)


@pytest.mark.parametrize(
    "response,expected",
    [
        # No guardrail configured: the key is simply absent. Must be False.
        ({"stopReason": "end_turn"}, False),
        ({}, False),
        # Explicitly not activated.
        ({"amazon-bedrock-guardrailAction": "NONE"}, False),
        # Genuine activations must still be detected.
        ({"amazon-bedrock-guardrailAction": "GUARDRAIL_INTERVENED"}, True),
        ({"stopReason": "guardrail_intervened"}, True),
        ({"results": [{"completionReason": "CONTENT_FILTERED"}]}, True),
        # A non-filtered result alongside no action key is not an activation.
        ({"results": [{"completionReason": "FINISH"}]}, False),
    ],
)
def test_is_guardrail_activated(response, expected):
    assert is_guardrail_activated(response) is expected


def test_converse_without_guardrail_does_not_record_activation():
    # Regression for #4471: a plain converse response (no guardrail action key,
    # no guardrail trace) must not touch the activation counter.
    metric_params = MagicMock()
    span = MagicMock()
    response = {"stopReason": "end_turn", "output": {"message": {"role": "assistant"}}}

    guardrail_converse(span, response, "aws", "amazon.titan-text", metric_params)

    metric_params.guardrail_activation.add.assert_not_called()


def test_converse_with_activation_records_once():
    metric_params = MagicMock()
    span = MagicMock()
    response = {"stopReason": "guardrail_intervened"}

    guardrail_converse(span, response, "aws", "amazon.titan-text", metric_params)

    metric_params.guardrail_activation.add.assert_called_once()


def test_activation_with_metrics_disabled_does_not_crash():
    # Regression for #4471 (second half): when metrics are disabled the SDK sets
    # guardrail_activation to None. A genuine activation must still set the span
    # attributes without raising AttributeError on the missing counter.
    metric_params = MagicMock()
    metric_params.guardrail_activation = None
    span = MagicMock()
    response = {"stopReason": "guardrail_intervened"}

    # Must not raise.
    guardrail_converse(span, response, "aws", "amazon.titan-text", metric_params)
