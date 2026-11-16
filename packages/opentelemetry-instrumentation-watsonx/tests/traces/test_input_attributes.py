import pytest
from opentelemetry.instrumentation.watsonx import _set_input_attributes
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture
def tracer_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(__name__), exporter


def _prompt_attributes(exporter):
    span = exporter.get_finished_spans()[-1]
    return {k: v for k, v in span.attributes.items() if k.startswith("gen_ai.prompt")}


def test_string_prompt_sets_prompt_attribute(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    with tracer.start_as_current_span("generate") as span:
        _set_input_attributes(span, kwargs={"prompt": " hello "})

    assert _prompt_attributes(exporter) == {"gen_ai.prompt.0.user": "hello"}


def test_list_prompt_sets_one_attribute_per_prompt(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    with tracer.start_as_current_span("generate") as span:
        _set_input_attributes(span, kwargs={"prompt": ["hello", "world"]})

    assert _prompt_attributes(exporter) == {
        "gen_ai.prompt.0.user": "hello",
        "gen_ai.prompt.1.user": "world",
    }


def test_list_prompt_passed_positionally(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    with tracer.start_as_current_span("generate") as span:
        _set_input_attributes(span, args=(["hello", "world"],))

    assert _prompt_attributes(exporter) == {
        "gen_ai.prompt.0.user": "hello",
        "gen_ai.prompt.1.user": "world",
    }
