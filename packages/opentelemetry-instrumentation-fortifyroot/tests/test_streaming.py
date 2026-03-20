import pytest

from opentelemetry.instrumentation.fortifyroot import (
    SafetyDecision,
    SafetyFinding,
    SafetyLocation,
    SafetyResult,
    clear_safety_handlers,
    create_completion_safety_stream,
    register_completion_safety_stream_factory,
)
from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

pytestmark = pytest.mark.fr


class _FakeStreamSession:
    def __init__(self, results):
        self._results = list(results)

    def process_chunk(self, text):
        return self._results.pop(0)

    def flush(self):
        return self._results.pop(0)


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


def test_create_completion_safety_stream_emits_events_and_returns_masked_text():
    exporter, tracer = _test_tracer()
    register_completion_safety_stream_factory(
        lambda _: _FakeStreamSession(
            [
                SafetyResult(
                    text="[PII.email]",
                    overall_action="mask",
                    findings=[
                        SafetyFinding(
                            category="pii",
                            severity="high",
                            action="mask",
                            rule_name="PII.email",
                            start=2,
                            end=15,
                        )
                    ],
                ),
                SafetyResult(text="tail", overall_action="allow", findings=[]),
            ]
        )
    )

    with tracer.start_as_current_span("test-span") as span:
        stream = create_completion_safety_stream(
            span=span,
            provider="OpenAI",
            span_name="openai.chat",
            location=SafetyLocation.COMPLETION,
            request_type="chat",
            segment_index=0,
            segment_role="assistant",
        )
        assert stream is not None
        assert stream.process_chunk("secret") == "[PII.email]"
        assert stream.flush() == "tail"

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert len(spans[0].events) == 1
    assert spans[0].events[0].attributes["fortifyroot.safety.category"] == "PII"
    assert spans[0].events[0].attributes["fortifyroot.safety.action"] == SafetyDecision.MASK.value


def test_create_completion_safety_stream_fails_open_when_factory_or_session_errors():
    _, tracer = _test_tracer()
    register_completion_safety_stream_factory(
        lambda _: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with tracer.start_as_current_span("test-span") as span:
        assert (
            create_completion_safety_stream(
                span=span,
                provider="OpenAI",
                span_name="openai.chat",
                location=SafetyLocation.COMPLETION,
            )
            is None
        )

    class _ExplodingSession:
        def process_chunk(self, text):
            raise RuntimeError("boom")

        def flush(self):
            raise RuntimeError("boom")

    register_completion_safety_stream_factory(lambda _: _ExplodingSession())

    with tracer.start_as_current_span("test-span-2") as span:
        stream = create_completion_safety_stream(
            span=span,
            provider="OpenAI",
            span_name="openai.chat",
            location=SafetyLocation.COMPLETION,
        )
        assert stream is not None
        assert stream.process_chunk("secret") == "secret"
        assert stream.flush() == ""


def test_create_completion_safety_stream_returns_none_without_factory_or_session():
    _, tracer = _test_tracer()

    with tracer.start_as_current_span("test-span") as span:
        assert (
            create_completion_safety_stream(
                span=span,
                provider="OpenAI",
                span_name="openai.chat",
                location=SafetyLocation.COMPLETION,
            )
            is None
        )

    register_completion_safety_stream_factory(lambda _: None)

    with tracer.start_as_current_span("test-span-2") as span:
        assert (
            create_completion_safety_stream(
                span=span,
                provider="OpenAI",
                span_name="openai.chat",
                location=SafetyLocation.COMPLETION,
            )
            is None
        )


def test_completion_text_stream_group_handles_empty_missing_and_flush_all(monkeypatch):
    created = []

    class _StubStream:
        def __init__(self, chunk_text, flush_text):
            self._chunk_text = chunk_text
            self._flush_text = flush_text

        def process_chunk(self, text):
            return self._chunk_text

        def flush(self):
            return self._flush_text

    stub_streams = [
        None,
        _StubStream("masked-a", "tail-a"),
        _StubStream("masked-b", ""),
    ]

    def _factory(**kwargs):
        created.append(kwargs)
        return stub_streams[len(created) - 1]

    monkeypatch.setattr(
        "opentelemetry.instrumentation.fortifyroot.text_streaming.create_completion_safety_stream",
        _factory,
    )

    group = CompletionTextStreamGroup(
        span=None,
        provider="OpenAI",
        span_name="openai.chat",
        request_type="chat",
    )

    assert group.process("empty", "", segment_index=0, segment_role="assistant") == ""
    assert group.process("no-stream", "raw", segment_index=0, segment_role="assistant") == "raw"
    assert group.process("masked-a", "secret-a", segment_index=1, segment_role="assistant") == "masked-a"
    assert group.process("masked-b", "secret-b", segment_index=2, segment_role="assistant") == "masked-b"
    assert group.flush("missing") == ""
    assert group.flush_all() == {"masked-a": "tail-a"}

    assert created[0]["metadata"] is None
    assert created[1]["segment_index"] == 1
    assert created[2]["segment_role"] == "assistant"
