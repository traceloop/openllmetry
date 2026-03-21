from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyFinding,
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
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
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


def _test_tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    return exporter, tracer


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


def test_build_streaming_response_emits_safety_events():
    class _EventfulSession:
        def process_chunk(self, text):
            return SafetyResult(
                text="[SECRET.output]",
                overall_action="MASK",
                findings=[
                    SafetyFinding(
                        category="SECRET",
                        severity="HIGH",
                        action="MASK",
                        rule_name="SECRET.output",
                        start=0,
                        end=len(text),
                    )
                ],
            )

        def flush(self):
            return SafetyResult(text="", overall_action="ALLOW", findings=[])

    exporter, tracer = _test_tracer()
    register_completion_safety_stream_factory(lambda _: _EventfulSession())
    handled = []

    response = [
        SimpleNamespace(
            id="resp-chat",
            model="demo",
            usage=None,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    delta=SimpleNamespace(content="secret"),
                )
            ],
        )
    ]

    with tracer.start_as_current_span("together.chat") as span:
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

    assert chunks[0].choices[0].delta.content == "[SECRET.output]"
    assert handled[0].choices[0].message.content == "[SECRET.output]"
    assert exporter.get_finished_spans()[0].events[0].name == "fortifyroot.safety.violation"


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
