from unittest.mock import MagicMock

import pytest
from opentelemetry.instrumentation.bedrock import (
    MetricParams,
    _handle_async_converse_stream,
    _handle_converse_stream,
)
from opentelemetry.instrumentation.bedrock.guardrail import (
    guardrail_converse,
    guardrail_handling,
    is_guardrail_activated,
)

MODEL_ID = "amazon.titan-text-express-v1"

GUARDRAIL_TRACE = {
    "guardrail": {
        "inputAssessment": {
            "gr-1:DRAFT": {
                "invocationMetrics": {
                    "guardrailProcessingLatency": 120,
                    "guardrailCoverage": {"textCharacters": {"guarded": 30, "total": 30}},
                },
                "sensitiveInformationPolicy": {"piiEntities": [{"type": "ADDRESS"}]},
            }
        }
    }
}


def _metric_params(enabled=True):
    return MetricParams(*[MagicMock() if enabled else None for _ in range(12)])


def _stream_events(stop_reason):
    # Bedrock sends stopReason on messageStop; the trailing metadata frame omits it.
    return [
        {"messageStart": {"role": "assistant"}},
        {"messageStop": {"stopReason": stop_reason}},
        {"metadata": {"trace": GUARDRAIL_TRACE}},
    ]


class _FakeStream:
    def __init__(self, events):
        self._events = list(events)

    def _parse_event(self):
        return self._events.pop(0)


class _FakeAsyncStream(_FakeStream):
    async def _parse_event(self):
        return self._events.pop(0)


def test_response_without_guardrail_key_is_not_an_activation():
    # Bedrock omits amazon-bedrock-guardrailAction when no guardrail is configured.
    assert is_guardrail_activated({"stopReason": "end_turn"}) is False
    assert is_guardrail_activated({}) is False


def test_configured_guardrail_that_did_not_fire_is_not_an_activation():
    assert is_guardrail_activated({"amazon-bedrock-guardrailAction": "NONE"}) is False


def test_activations_are_detected():
    assert is_guardrail_activated({"amazon-bedrock-guardrailAction": "INTERVENED"}) is True
    assert is_guardrail_activated({"stopReason": "guardrail_intervened"}) is True
    assert (
        is_guardrail_activated({"results": [{"completionReason": "CONTENT_FILTERED"}]})
        is True
    )


def test_converse_without_guardrail_does_not_record_activation():
    metric_params = _metric_params()
    guardrail_converse(MagicMock(), {"stopReason": "end_turn"}, "aws", "m", metric_params)
    metric_params.guardrail_activation.add.assert_not_called()


def test_converse_activation_is_recorded():
    metric_params = _metric_params()
    guardrail_converse(
        MagicMock(),
        {"stopReason": "guardrail_intervened", "trace": GUARDRAIL_TRACE},
        "aws",
        "m",
        metric_params,
    )
    metric_params.guardrail_activation.add.assert_called_once()


def test_activation_with_metrics_disabled_still_sets_span_attributes():
    converse_span = MagicMock()
    guardrail_converse(
        converse_span,
        {"stopReason": "guardrail_intervened", "trace": GUARDRAIL_TRACE},
        "aws",
        "m",
        _metric_params(enabled=False),
    )
    assert converse_span.set_attribute.called

    invoke_span = MagicMock()
    guardrail_handling(
        invoke_span,
        {
            "amazon-bedrock-guardrailAction": "INTERVENED",
            "amazon-bedrock-trace": {
                "guardrail": {"input": GUARDRAIL_TRACE["guardrail"]["inputAssessment"]}
            },
        },
        "aws",
        "m",
        _metric_params(enabled=False),
    )
    assert invoke_span.set_attribute.called


@pytest.mark.parametrize(
    "stop_reason,activations", [("guardrail_intervened", 1), ("end_turn", 0)]
)
def test_converse_stream_reads_stop_reason_from_message_stop(stop_reason, activations):
    metric_params = _metric_params()
    stream = _FakeStream(_stream_events(stop_reason))
    _handle_converse_stream(
        MagicMock(), {"modelId": MODEL_ID}, {"stream": stream}, metric_params, None
    )

    for _ in range(3):
        stream._parse_event()

    assert metric_params.guardrail_activation.add.call_count == activations


@pytest.mark.parametrize(
    "stop_reason,activations", [("guardrail_intervened", 1), ("end_turn", 0)]
)
async def test_async_converse_stream_reads_stop_reason_from_message_stop(
    stop_reason, activations
):
    metric_params = _metric_params()
    response = {"stream": _FakeAsyncStream(_stream_events(stop_reason))}
    _handle_async_converse_stream(
        MagicMock(), {"modelId": MODEL_ID}, response, metric_params, None
    )

    for _ in range(3):
        await response["stream"]._parse_event()

    assert metric_params.guardrail_activation.add.call_count == activations


def test_converse_stream_activation_with_metrics_disabled_ends_span():
    span = MagicMock()
    stream = _FakeStream(_stream_events("guardrail_intervened"))
    _handle_converse_stream(
        span, {"modelId": MODEL_ID}, {"stream": stream}, _metric_params(enabled=False), None
    )

    for _ in range(3):
        stream._parse_event()

    span.end.assert_called_once()
