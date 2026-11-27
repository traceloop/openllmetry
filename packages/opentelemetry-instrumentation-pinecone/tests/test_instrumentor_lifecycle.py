"""Instrumentor lifecycle tests: metrics disabled, and wrappers outliving uninstrument()."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import opentelemetry.instrumentation.pinecone as pinecone_instrumentation
from opentelemetry.instrumentation.pinecone import (
    PineconeInstrumentor,
    _set_response_attributes,
    _wrap,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SpanAttributes


def test_instrument_with_metrics_disabled(monkeypatch):
    wrap_function_wrapper = MagicMock()
    monkeypatch.setattr(pinecone_instrumentation, "is_metrics_enabled", lambda: False)
    monkeypatch.setattr(pinecone_instrumentation, "wrap_function_wrapper", wrap_function_wrapper)
    # keep the session-wide instrumentation generation untouched
    monkeypatch.setitem(pinecone_instrumentation._state, "generation", pinecone_instrumentation._state["generation"])

    PineconeInstrumentor()._instrument(tracer_provider=TracerProvider())

    assert wrap_function_wrapper.called


def test_usage_span_attributes_without_metrics():
    span = MagicMock()

    _set_response_attributes(span, None, None, {}, {"usage": {"read_units": 3}})

    span.set_attribute.assert_any_call(SpanAttributes.PINECONE_USAGE_READ_UNITS, 3)
    span.set_attribute.assert_any_call(SpanAttributes.PINECONE_USAGE_WRITE_UNITS, 0)


def test_wrapper_from_previous_generation_passes_through(monkeypatch):
    # pinecone 10 Index objects keep the wrapped method they saw at construction,
    # so a wrapper can still be called after uninstrument().
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)

    monkeypatch.setitem(pinecone_instrumentation._state, "generation", pinecone_instrumentation._state["generation"])
    wrapper = _wrap(tracer, None, None, None, None, {"method": "query", "span_name": "pinecone.query"})
    instance = SimpleNamespace(_config=SimpleNamespace(host="http://localhost"))
    wrapped = MagicMock(return_value={"matches": []})

    assert wrapper(wrapped, instance, (), {"top_k": 1}) == {"matches": []}
    assert [s.name for s in exporter.get_finished_spans()] == ["pinecone.query"]

    # simulate uninstrument()
    next_generation = pinecone_instrumentation._state["generation"] + 1
    monkeypatch.setitem(pinecone_instrumentation._state, "generation", next_generation)
    exporter.clear()

    assert wrapper(wrapped, instance, (), {"top_k": 1}) == {"matches": []}
    assert exporter.get_finished_spans() == ()
    assert wrapped.call_count == 2
