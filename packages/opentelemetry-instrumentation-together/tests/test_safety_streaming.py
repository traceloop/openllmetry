from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyResult,
    clear_completion_safety_stream_factory,
    clear_safety_handlers,
    register_completion_safety_stream_factory,
)
from opentelemetry.instrumentation.together.streaming_safety import (
    build_async_streaming_response,
    build_streaming_response,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv_ai import LLMRequestTypeValues

pytestmark = pytest.mark.fr


def setup_function():
    clear_safety_handlers()
    clear_completion_safety_stream_factory()


def teardown_function():
    clear_safety_handlers()
    clear_completion_safety_stream_factory()


def _test_span():
    provider = TracerProvider()
    tracer = provider.get_tracer(__name__)
    return tracer.start_span("together.chat")


class _FakeStreamSession:
    def __init__(self, results=None, flush_result=""):
        self._results = list(results or [])
        self._flush_result = flush_result

    def process_chunk(self, text):
        if self._results:
            return SafetyResult(text=self._results.pop(0), overall_action="MASK", findings=[])
        return SafetyResult(text=text, overall_action="ALLOW", findings=[])

    def flush(self):
        return SafetyResult(text=self._flush_result, overall_action="ALLOW", findings=[])


def test_build_streaming_response_masks_chat_chunks_and_builds_final_response():
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a"], flush_result="tail")
    )
    span = _test_span()
    handled = []

    response = [
        SimpleNamespace(
            id="resp-chat",
            model="demo",
            usage=None,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason=None,
                    delta=SimpleNamespace(content="a"),
                )
            ],
        )
    ]

    try:
        chunks = list(
            build_streaming_response(
                response,
                span=span,
                event_logger=None,
                llm_request_type=LLMRequestTypeValues.CHAT,
                span_name="together.chat",
                handle_response=lambda _span, _logger, _type, final: handled.append(final),
            )
        )
    finally:
        if span.is_recording():
            span.end()

    assert chunks[0].choices[0].delta.content == "masked-atail"
    assert handled[0].choices[0].message.content == "masked-atail"


@pytest.mark.asyncio
async def test_build_async_streaming_response_masks_completion_chunks():
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a"], flush_result="tail")
    )
    span = _test_span()
    handled = []

    async def _response():
        yield SimpleNamespace(
            id="resp-completion",
            model="demo",
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason=None,
                    delta=SimpleNamespace(content="a"),
                )
            ],
        )

    chunks = []
    try:
        async for chunk in build_async_streaming_response(
            _response(),
            span=span,
            event_logger=None,
            llm_request_type=LLMRequestTypeValues.COMPLETION,
            span_name="together.completion",
            handle_response=lambda _span, _logger, _type, final: handled.append(final),
        ):
            chunks.append(chunk)
    finally:
        if span.is_recording():
            span.end()

    assert chunks[0].choices[0].delta.content == "masked-atail"
    assert handled[0].choices[0].text == "masked-atail"
