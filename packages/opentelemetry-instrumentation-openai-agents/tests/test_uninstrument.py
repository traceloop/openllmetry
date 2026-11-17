"""Tests for uninstrument() deregistering the OpenTelemetry trace processor.

Regression tests for https://github.com/traceloop/openllmetry/issues/4634:
uninstrument() only unwrapped the realtime session wrappers and left the
OpenTelemetryTracingProcessor registered with the Agents SDK, so runs stayed
traced and every instrument()/uninstrument() cycle added another processor.

Note: BaseInstrumentor is a singleton, and the session fixtures in conftest.py
keep one shared instrumentor instrumented for the whole test session. These
tests uninstrument that shared instance and restore the instrumented state on
teardown so later modules still see it.
"""

import pytest
from unittest.mock import MagicMock

from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor


def _registered_processors():
    """Return the processors currently registered with the Agents SDK provider."""
    from agents.tracing.setup import get_trace_provider

    provider = get_trace_provider()
    multi_processor = getattr(provider, "_multi_processor", None)
    if multi_processor is None:
        pytest.fail("Agents SDK default trace provider internals not found")
    return list(getattr(multi_processor, "_processors", ()))


def _drop_processor(processor):
    """Remove a leftover processor added by a test, keeping everything else."""
    from agents import set_trace_processors

    set_trace_processors([p for p in _registered_processors() if p is not processor])


@pytest.fixture
def instrumentor():
    inst = OpenAIAgentsInstrumentor()
    if not inst.is_instrumented_by_opentelemetry:
        inst.instrument()
    yield inst
    # Restore the session-wide instrumented state expected by other modules
    if inst.is_instrumented_by_opentelemetry:
        inst.uninstrument()
    OpenAIAgentsInstrumentor().instrument()


def test_uninstrument_removes_registered_processor(instrumentor):
    processor = getattr(instrumentor, "_otel_processor", None)
    assert processor is not None
    assert processor in _registered_processors()

    instrumentor.uninstrument()

    assert processor not in _registered_processors()
    assert instrumentor._otel_processor is None


def test_uninstrument_keeps_other_processors(instrumentor):
    other = MagicMock()
    from agents import add_trace_processor

    add_trace_processor(other)
    try:
        processor = instrumentor._otel_processor

        instrumentor.uninstrument()

        processors = _registered_processors()
        assert processor not in processors
        assert other in processors
    finally:
        _drop_processor(other)


def test_reinstrument_does_not_duplicate_processor(instrumentor):
    from opentelemetry.instrumentation.openai_agents._hooks import (
        OpenTelemetryTracingProcessor,
    )

    instrumentor.uninstrument()
    instrumentor.instrument()
    instrumentor.instrument()  # no-op per BaseInstrumentor's already-instrumented guard

    ours = [
        p
        for p in _registered_processors()
        if isinstance(p, OpenTelemetryTracingProcessor)
    ]
    assert len(ours) == 1


def test_uninstrument_restores_replaced_processors():
    prior = MagicMock()
    from agents import add_trace_processor

    add_trace_processor(prior)
    inst = OpenAIAgentsInstrumentor(replace_existing_processors=True)
    try:
        # The shared singleton is already instrumented by the session fixtures;
        # uninstrument so the replace_existing_processors path actually runs.
        if inst.is_instrumented_by_opentelemetry:
            inst.uninstrument()
        inst.instrument()
        processors = _registered_processors()
        assert inst._otel_processor in processors
        assert prior not in processors

        inst.uninstrument()

        processors = _registered_processors()
        assert inst._otel_processor is None
        assert prior in processors
    finally:
        if inst.is_instrumented_by_opentelemetry:
            inst.uninstrument()
        OpenAIAgentsInstrumentor().instrument()
        _drop_processor(prior)
