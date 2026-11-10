"""Opt-in delegation / authority attributes on agent and handoff spans."""

import pytest
from opentelemetry import context as context_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# Same key the opentelemetry.semconv_ai.delegation helper writes to. Set directly so
# the test also runs against semconv-ai releases that predate the helper.
DELEGATION_KEY = "traceloop.delegation"
DELEGATION = {
    "traceloop.delegation.root_principal": "user:42",
    "traceloop.delegation.chain_id": "dlg_7f3a",
    "traceloop.delegation.depth": 1,
    "traceloop.delegation.scope_hash": "sha256:" + "ab" * 32,
}


class MockSpan:
    def __init__(self, span_data, trace_id="t1"):
        self.span_data = span_data
        self.trace_id = trace_id
        self.error = None


@pytest.fixture
def processor_and_exporter():
    from opentelemetry.instrumentation.openai_agents._hooks import OpenTelemetryTracingProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OpenTelemetryTracingProcessor(provider.get_tracer("test-delegation")), exporter


@pytest.fixture
def delegation_context():
    token = context_api.attach(context_api.set_value(DELEGATION_KEY, dict(DELEGATION)))
    yield
    context_api.detach(token)


def _delegation_attrs(span):
    return {k: v for k, v in span.attributes.items() if k.startswith("traceloop.delegation.")}


def _run_handoff_flow(processor):
    from agents import AgentSpanData, FunctionSpanData, HandoffSpanData

    triage = MockSpan(AgentSpanData(name="Triage", handoffs=["Billing"], tools=[], output_type=""))
    handoff = MockSpan(HandoffSpanData(from_agent="Triage", to_agent="Billing"))
    billing = MockSpan(AgentSpanData(name="Billing", handoffs=[], tools=[], output_type=""))
    tool = MockSpan(FunctionSpanData(name="refund", input="{}", output="ok"))
    for span in (triage, handoff):
        processor.on_span_start(span)
    processor.on_span_end(handoff)
    processor.on_span_end(triage)
    processor.on_span_start(billing)
    processor.on_span_start(tool)
    processor.on_span_end(tool)
    processor.on_span_end(billing)


def test_agent_and_handoff_spans_record_delegation(processor_and_exporter, delegation_context):
    processor, exporter = processor_and_exporter
    _run_handoff_flow(processor)

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert _delegation_attrs(spans["Triage.agent"]) == DELEGATION
    assert _delegation_attrs(spans["Billing.agent"]) == DELEGATION
    handoff = spans["Triage → Billing.handoff"]
    assert _delegation_attrs(handoff) == DELEGATION
    assert handoff.attributes["gen_ai.handoff.from_agent"] == "Triage"
    assert handoff.attributes["gen_ai.handoff.to_agent"] == "Billing"
    # Tool spans aren't stamped; they inherit the authority from their agent span.
    assert not _delegation_attrs(spans["refund.tool"])


def test_no_delegation_attributes_without_opt_in(processor_and_exporter):
    processor, exporter = processor_and_exporter
    _run_handoff_flow(processor)

    spans = exporter.get_finished_spans()
    assert len(spans) == 4
    assert all(not _delegation_attrs(s) for s in spans)


def test_foreign_or_malformed_values_are_ignored(processor_and_exporter):
    from agents import AgentSpanData

    processor, exporter = processor_and_exporter
    value = {
        "traceloop.delegation.root_principal": "user:1",
        "authorization": "Bearer secret",
        "traceloop.delegation.depth": False,
        "traceloop.delegation.chain_id": ["x"],
    }
    token = context_api.attach(context_api.set_value(DELEGATION_KEY, value))
    try:
        span = MockSpan(AgentSpanData(name="A", handoffs=[], tools=[], output_type=""))
        processor.on_span_start(span)
        processor.on_span_end(span)
    finally:
        context_api.detach(token)

    (agent_span,) = exporter.get_finished_spans()
    assert _delegation_attrs(agent_span) == {"traceloop.delegation.root_principal": "user:1"}
    assert "authorization" not in agent_span.attributes


def test_non_mapping_value_is_ignored(processor_and_exporter):
    from agents import AgentSpanData

    processor, exporter = processor_and_exporter
    token = context_api.attach(context_api.set_value(DELEGATION_KEY, "user:1"))
    try:
        span = MockSpan(AgentSpanData(name="A", handoffs=[], tools=[], output_type=""))
        processor.on_span_start(span)
        processor.on_span_end(span)
    finally:
        context_api.detach(token)

    (agent_span,) = exporter.get_finished_spans()
    assert not _delegation_attrs(agent_span)
