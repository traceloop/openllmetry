"""Lifecycle tests: uninstrument() must detach the wrappers it installed."""

import together

from opentelemetry.instrumentation.together import TogetherAiInstrumentor

CHAT_CLASS = together.resources.chat.completions.ChatCompletions


def _call_chat_create():
    """Call the instrumented method with a stand-in client.

    The wrapper opens the span before delegating, so a traced call records a
    finished span even though the stub client raises.
    """
    try:
        CHAT_CLASS.create(object(), model="meta-llama/Llama-3-8b")
    except Exception:
        pass


def test_uninstrument_detaches_the_chat_wrapper(span_exporter, tracer_provider):
    original_create = CHAT_CLASS.__dict__["create"]
    instrumentor = TogetherAiInstrumentor()

    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        _call_chat_create()
        assert [span.name for span in span_exporter.get_finished_spans()] == [
            "together.chat"
        ]
    finally:
        instrumentor.uninstrument()

    assert CHAT_CLASS.__dict__["create"] is original_create

    span_exporter.clear()
    _call_chat_create()
    assert span_exporter.get_finished_spans() == ()
