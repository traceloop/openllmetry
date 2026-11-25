"""Regression tests for openllmetry#4635.

With two concurrent runs of the same agent, the handoff span of one run was
recorded in the other run's trace: ``_start_handoff_span`` parented the handoff
to ``_find_agent_span(from_agent)``, which searched the open agent spans of ALL
runs by agent name and returned the first match. The lookup is now scoped to
the handoff's own trace id.
"""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from agents import AgentSpanData, HandoffSpanData
from opentelemetry.instrumentation.openai_agents._hooks import (
    OpenTelemetryTracingProcessor,
)


@pytest.fixture
def tracer_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test-handoff-parenting"), exporter


@pytest.fixture
def processor(tracer_and_exporter):
    tracer, _ = tracer_and_exporter
    return OpenTelemetryTracingProcessor(tracer)


class MockAgentSpan:
    """Minimal hashable mock of an Agents SDK span object."""

    def __init__(self, span_data, trace_id, error=None):
        self.span_data = span_data
        self.trace_id = trace_id
        self.error = error


def _start_run_agent_span(processor, agent_name, trace_id):
    """Register an open agent span for one run, like on_span_start would."""
    sdk_span = MockAgentSpan(
        AgentSpanData(name=agent_name, handoffs=[], tools=[], output_type=""),
        trace_id,
    )
    otel_span, _name_token = processor._start_agent_span(sdk_span.span_data, None, trace_id)
    processor._otel_spans[sdk_span] = otel_span
    return otel_span


class TestHandoffSpanParenting:
    def test_handoff_span_parented_to_own_run_agent_span(self, tracer_and_exporter, processor):
        """Two concurrent runs of the same agent: run B's handoff must land
        in run B's trace, parented to run B's agent span (was: run A's)."""
        otel_a = _start_run_agent_span(processor, "Assistant", "trace-A")
        otel_b = _start_run_agent_span(processor, "Assistant", "trace-B")

        handoff = processor._start_handoff_span(
            HandoffSpanData(from_agent="Assistant", to_agent="Support"),
            None,
            "trace-B",
        )

        assert handoff.parent is not None
        assert handoff.parent.span_id == otel_b.get_span_context().span_id, (
            "handoff span must be parented to its own run's agent span"
        )
        assert handoff.parent.span_id != otel_a.get_span_context().span_id

        handoff.end()
        otel_a.end()
        otel_b.end()

    def test_handoff_span_single_run_still_parented(self, processor):
        """The common single-run case keeps working after the scoping fix."""
        otel_span = _start_run_agent_span(processor, "Assistant", "trace-A")

        handoff = processor._start_handoff_span(
            HandoffSpanData(from_agent="Assistant", to_agent="Support"),
            None,
            "trace-A",
        )

        assert handoff.parent is not None
        assert handoff.parent.span_id == otel_span.get_span_context().span_id

        handoff.end()
        otel_span.end()

    def test_handoff_span_unknown_from_agent_keeps_fallback_context(self, processor):
        """No matching agent span: the passed parent context is left alone."""
        otel_span = _start_run_agent_span(processor, "Assistant", "trace-A")

        handoff = processor._start_handoff_span(
            HandoffSpanData(from_agent="Nobody", to_agent="Support"),
            None,
            "trace-A",
        )

        assert handoff.parent is None  # root span, unchanged fallback

        handoff.end()
        otel_span.end()

    def test_find_agent_span_without_trace_id_falls_back_to_name(self, processor):
        """Callers without a trace id keep the previous name-only lookup."""
        otel_a = _start_run_agent_span(processor, "Assistant", "trace-A")

        assert processor._find_agent_span("Assistant") is otel_a
        assert processor._find_agent_span("Missing") is None

        otel_a.end()
