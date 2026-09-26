"""Regression test: async streaming must record the model's text.

_build_from_streaming_response accumulates the streamed chunks into
complete_response and hands that to handle_streaming_response. Its async twin
accumulates the same string and then hands over the (now exhausted) async
generator instead, so gen_ai.completion.0.content ends up holding the
generator's repr rather than the answer.
"""

import pytest
from unittest.mock import Mock

from opentelemetry.instrumentation.vertexai import (
    _abuild_from_streaming_response,
    _build_from_streaming_response,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes

CONTENT_KEY = f"{gen_ai_attributes.GEN_AI_COMPLETION}.0.content"


class _Chunk:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = None


def _recording_span():
    span = Mock()
    span.is_recording.return_value = True
    captured = {}
    span.set_attribute = lambda key, value: captured.__setitem__(key, value)
    return span, captured


def test_sync_streaming_records_the_text():
    span, captured = _recording_span()

    list(
        _build_from_streaming_response(
            span, None, iter([_Chunk("Hello"), _Chunk(" world")]), "gemini-pro"
        )
    )

    assert captured[CONTENT_KEY] == "Hello world"


@pytest.mark.asyncio
async def test_async_streaming_records_the_text():
    span, captured = _recording_span()

    async def chunks():
        for chunk in (_Chunk("Hello"), _Chunk(" world")):
            yield chunk

    async for _ in _abuild_from_streaming_response(
        span, None, chunks(), "gemini-pro"
    ):
        pass

    assert captured[CONTENT_KEY] == "Hello world"
