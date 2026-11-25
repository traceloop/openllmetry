from types import SimpleNamespace
from unittest.mock import patch

import pytest
from opentelemetry.instrumentation.vertexai import _abuild_from_streaming_response
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


class RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.statuses: list[object] = []
        self.ended = False

    def is_recording(self) -> bool:
        return True

    def set_attribute(self, name: str, value: object) -> None:
        self.attributes[name] = value

    def set_status(self, status: object) -> None:
        self.statuses.append(status)

    def end(self) -> None:
        self.ended = True


def _chunk(
    text: str,
    *,
    usage_metadata: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(text=text, usage_metadata=usage_metadata)


async def _stream_chunks():
    usage_metadata = SimpleNamespace(
        total_token_count=7,
        candidates_token_count=4,
        prompt_token_count=3,
        cached_content_token_count=None,
    )
    yield _chunk("hello ")
    yield _chunk("world", usage_metadata=usage_metadata)


@pytest.mark.asyncio
async def test_async_streaming_records_accumulated_text_after_drain() -> None:
    span = RecordingSpan()

    with (
        patch("opentelemetry.instrumentation.vertexai.should_emit_events", return_value=False),
        patch("opentelemetry.instrumentation.vertexai.span_utils.should_send_prompts", return_value=True),
    ):
        response = _abuild_from_streaming_response(
            span=span,
            event_logger=None,
            response=_stream_chunks(),
            llm_model="gemini-2.5-pro",
        )

        observed = [chunk.text async for chunk in response]

    assert observed == ["hello ", "world"]
    assert (
        span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"] == "hello world"
    )
    assert span.ended is True
