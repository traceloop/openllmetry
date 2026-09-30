from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider

from traceloop.sdk.propagation import extract_trace_context, inject_trace_context


def test_inject_and_extract_trace_context_round_trip():
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    carrier = {}

    with tracer.start_as_current_span("parent") as span:
        inject_trace_context(carrier)
        extracted = extract_trace_context(carrier)
        extracted_span = trace.get_current_span(extracted)

        assert carrier["traceparent"].startswith("00-")
        assert f"-{span.get_span_context().span_id:016x}-" in carrier["traceparent"]
        assert extracted_span.get_span_context().trace_id == span.get_span_context().trace_id


def test_inject_accepts_an_existing_carrier():
    carrier = {"x-request-id": "request-1"}
    result = inject_trace_context(carrier)

    assert result is carrier
    assert result["x-request-id"] == "request-1"


def test_extract_does_not_mutate_current_context():
    carrier = {}
    current = context.get_current()

    extracted = extract_trace_context(carrier)

    assert extracted is not current
