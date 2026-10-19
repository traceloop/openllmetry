"""Converse / ConverseStream token metrics must be labeled with the called model.

metric_params is one object shared by every call on an instrumentor, so a
handler that does not set vendor/model before recording reuses whatever the
previous invoke_model call left there (or "" on a fresh instrumentor).
"""

from unittest.mock import MagicMock

from opentelemetry.instrumentation.bedrock import (
    _handle_async_converse_stream,
    _handle_converse,
    _handle_converse_stream,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

USAGE = {"inputTokens": 11, "outputTokens": 5, "totalTokens": 16}
STREAM_EVENTS = [
    {"messageStart": {"role": "assistant"}},
    {"contentBlockDelta": {"delta": {"text": "pong"}}},
    {"messageStop": {"stopReason": "end_turn"}},
    {"metadata": {"usage": USAGE}},
]


def _metric_params_after_invoke_model():
    """metric_params as left behind by an earlier invoke_model call."""
    mp = MagicMock()
    mp.vendor = "aws.bedrock"
    mp.model = "claude-haiku-4-5-20251001-v1:0"
    mp.is_stream = False
    mp.duration_histogram = None
    mp.token_histogram = MagicMock()
    return mp


def _recorded_models(metric_params):
    return {
        call.kwargs["attributes"][GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        for call in metric_params.token_histogram.record.call_args_list
    }


class _Stream:
    def __init__(self, events):
        self._events = iter(events)

    def _parse_event(self):
        return next(self._events)


class _AsyncStream:
    def __init__(self, events):
        self._events = iter(events)

    async def _parse_event(self):
        return next(self._events)


class TestConverseMetricModelLabel:
    def test_converse(self):
        mp = _metric_params_after_invoke_model()
        response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "pong"}]}},
            "stopReason": "end_turn",
            "usage": USAGE,
        }
        _handle_converse(MagicMock(), {"modelId": "us.openai.gpt-6-sol"}, response, mp, None)
        assert _recorded_models(mp) == {"gpt-6-sol"}
        assert mp.is_stream is False

    def test_converse_stream(self):
        mp = _metric_params_after_invoke_model()
        stream = _Stream(STREAM_EVENTS)
        _handle_converse_stream(
            MagicMock(), {"modelId": "us.openai.gpt-6-luna"}, {"stream": stream}, mp, None
        )
        for _ in STREAM_EVENTS:
            stream._parse_event()
        assert _recorded_models(mp) == {"gpt-6-luna"}
        assert mp.is_stream is True

    async def test_async_converse_stream(self):
        mp = _metric_params_after_invoke_model()
        stream = _AsyncStream(STREAM_EVENTS)
        _handle_async_converse_stream(
            MagicMock(), {"modelId": "us.openai.gpt-6-luna"}, {"stream": stream}, mp, None
        )
        for _ in STREAM_EVENTS:
            await stream._parse_event()
        assert _recorded_models(mp) == {"gpt-6-luna"}
        assert mp.is_stream is True
