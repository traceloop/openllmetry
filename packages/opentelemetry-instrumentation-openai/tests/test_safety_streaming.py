from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyResult,
    clear_safety_handlers,
    register_completion_safety_stream_factory,
)
from opentelemetry.instrumentation.openai.shared.streaming_safety import (
    OpenAIChatStreamingSafety,
    OpenAICompletionStreamingSafety,
    _ensure_delta,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

pytestmark = pytest.mark.fr


class _FakeStreamSession:
    def __init__(self, process_results, flush_result=""):
        self._process_results = list(process_results)
        self._flush_result = flush_result

    def process_chunk(self, text):
        value = self._process_results.pop(0)
        return SafetyResult(text=value, overall_action="allow", findings=[])

    def flush(self):
        return SafetyResult(text=self._flush_result, overall_action="allow", findings=[])


def setup_function():
    clear_safety_handlers()


def teardown_function():
    clear_safety_handlers()


def _test_tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    return exporter, tracer


def test_chat_streaming_safety_masks_delta_content_and_flushes_on_finish():
    _, tracer = _test_tracer()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a"], flush_result="tail")
    )

    chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(content="secret"),
                finish_reason=None,
            )
        ]
    )
    final_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(),
                finish_reason="stop",
            )
        ]
    )

    with tracer.start_as_current_span("openai.chat") as span:
        helper = OpenAIChatStreamingSafety(span, "openai.chat")
        helper.process_chunk(chunk)
        helper.process_chunk(final_chunk)

    assert chunk.choices[0].delta.content == "masked-a"
    assert final_chunk.choices[0].delta.content == "tail"


def test_completion_streaming_safety_masks_text_and_flushes_on_finish():
    _, tracer = _test_tracer()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(["masked-a"], flush_result="tail")
    )

    chunk = SimpleNamespace(
        choices=[SimpleNamespace(index=0, text="secret", finish_reason=None)]
    )
    final_chunk = SimpleNamespace(
        choices=[SimpleNamespace(index=0, finish_reason="stop")]
    )

    with tracer.start_as_current_span("openai.completion") as span:
        helper = OpenAICompletionStreamingSafety(span, "openai.completion")
        helper.process_chunk(chunk)
        helper.process_chunk(final_chunk)

    assert chunk.choices[0].text == "masked-a"
    assert final_chunk.choices[0].text == "tail"


def test_ensure_delta_creates_mapping_delta():
    choice = {"index": 0}

    delta = _ensure_delta(choice)

    assert delta == {}
    assert choice["delta"] == {}


def test_ensure_delta_returns_none_when_choice_cannot_store_delta():
    class _NoAttrs:
        __slots__ = ()

    assert _ensure_delta(_NoAttrs()) is None
