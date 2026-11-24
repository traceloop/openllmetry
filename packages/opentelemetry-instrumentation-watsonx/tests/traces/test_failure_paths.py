from types import SimpleNamespace

import pytest
from opentelemetry.instrumentation.watsonx import _wrap, _build_and_set_stream_response
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

GENERATE = {"method": "generate", "span_name": "watsonx.generate"}
GENERATE_STREAM = {"method": "generate_text_stream", "span_name": "watsonx.generate_text_stream"}


@pytest.fixture(name="exporter")
def fixture_exporter():
    return InMemorySpanExporter()


@pytest.fixture(name="tracer")
def fixture_tracer(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(__name__)


@pytest.fixture(name="model")
def fixture_model():
    return SimpleNamespace(model_id="google/flan-ul2", params=None)


def _chunk(text):
    return {
        "model_id": "google/flan-ul2",
        "results": [
            {
                "generated_text": text,
                "input_token_count": 3,
                "generated_token_count": 1,
                "stop_reason": "not_finished",
            }
        ],
    }


def test_request_attributes_are_recorded(tracer, exporter, model):
    wrapper = _wrap(tracer, GENERATE, None, None, None, None, None)

    def fake_generate(*args, **kwargs):
        return {"model_id": "google/flan-ul2", "results": [{"generated_text": "2"}]}

    wrapper(fake_generate, model, (), {"prompt": "What is 1 + 1?"})

    (span,) = exporter.get_finished_spans()
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "google/flan-ul2"
    assert span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.user"] == "What is 1 + 1?"


def test_span_ends_when_the_call_fails(tracer, exporter, model):
    wrapper = _wrap(tracer, GENERATE, None, None, None, None, None)

    def failing_generate(*args, **kwargs):
        raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        wrapper(failing_generate, model, (), {"prompt": "hi"})

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code.name == "ERROR"
    assert span.attributes["error.type"] == "ConnectionError"


def test_span_ends_when_the_stream_fails(tracer, exporter):
    span = tracer.start_span("watsonx.generate_text_stream")

    def failing_stream():
        yield _chunk("Hello")
        raise ConnectionError("stream dropped")

    with pytest.raises(ConnectionError):
        list(_build_and_set_stream_response(span, None, failing_stream(), False, None, None, None, None))

    (finished,) = exporter.get_finished_spans()
    assert finished.name == "watsonx.generate_text_stream"
    assert finished.status.status_code.name == "ERROR"
    assert finished.attributes["error.type"] == "ConnectionError"


def test_span_ends_when_the_stream_is_abandoned(tracer, exporter):
    span = tracer.start_span("watsonx.generate_text_stream")

    stream = _build_and_set_stream_response(
        span, None, iter([_chunk("Hello"), _chunk(" world")]), False, None, None, None, None
    )
    assert next(stream) == "Hello"
    stream.close()

    assert [s.name for s in exporter.get_finished_spans()] == ["watsonx.generate_text_stream"]
