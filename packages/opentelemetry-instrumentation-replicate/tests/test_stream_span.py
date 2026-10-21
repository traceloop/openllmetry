import pytest
from opentelemetry.instrumentation.replicate import _build_from_streaming_response


def _failing_stream():
    yield "Hello"
    raise ConnectionError("stream dropped")


def _long_stream():
    yield "Hello"
    yield " world"


def test_stream_ends_span_when_stream_fails(tracer_provider, span_exporter):
    span = tracer_provider.get_tracer(__name__).start_span("replicate.stream")

    with pytest.raises(ConnectionError):
        list(_build_from_streaming_response(span, None, _failing_stream()))

    (finished,) = span_exporter.get_finished_spans()
    assert finished.name == "replicate.stream"
    assert finished.status.status_code.name == "ERROR"
    assert [e.name for e in finished.events] == ["exception"]


def test_stream_ends_span_when_abandoned(tracer_provider, span_exporter):
    span = tracer_provider.get_tracer(__name__).start_span("replicate.stream")

    stream = _build_from_streaming_response(span, None, _long_stream())
    assert next(stream) == "Hello"
    stream.close()

    assert [s.name for s in span_exporter.get_finished_spans()] == ["replicate.stream"]
