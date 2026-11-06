"""Smoke test for the Aleph Alpha tracing demo, run fully offline."""

import pytest
from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes

from sample_app import alephalpha_tracing_demo as demo


@pytest.fixture
def exporter(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(demo, "tracer", provider.get_tracer("test"))

    instrumentor = AlephAlphaInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    yield exporter
    instrumentor.uninstrument()


def test_ask_creates_completion_span_under_app_span(exporter):
    client = demo.make_client(offline=True, host="https://api.aleph-alpha.com")

    answer = demo.ask(client, "What is a trace?", "luminous-base")

    assert answer.startswith("OpenTelemetry is like a diary")

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {"ask_question", "alephalpha.completion"}

    app_span = spans["ask_question"]
    llm_span = spans["alephalpha.completion"]
    assert llm_span.parent.span_id == app_span.context.span_id
    assert llm_span.context.trace_id == app_span.context.trace_id

    assert llm_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "luminous-base"
    assert llm_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 18
    assert llm_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 27
    assert app_span.attributes["app.answer.length"] == len(answer)


def test_print_trace_summary_nests_llm_span(exporter, capsys):
    client = demo.make_client(offline=True, host="https://api.aleph-alpha.com")
    demo.ask(client, "What is a trace?", "luminous-base")

    demo.print_trace_summary(exporter.get_finished_spans())

    output = capsys.readouterr().out
    assert "- ask_question" in output
    assert "    - alephalpha.completion" in output
