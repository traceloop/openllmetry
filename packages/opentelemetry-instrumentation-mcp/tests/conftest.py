"""Test configuration module."""

import pytest
from opentelemetry.instrumentation.mcp import McpInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture(scope="session", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="session", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    yield provider
    provider.shutdown()


# Session-scoped: McpInstrumentor wraps process-level imports (BaseSession.send_request,
# post-import hooks on fastmcp.client, etc.). Re-instrumenting per test stacks wrappers
# because _uninstrument doesn't fully tear them down — that breaks tests whose assertions
# depend on a single wrapping layer (e.g. trace-id sharing in test_fastmcp.py).
@pytest.fixture(scope="session", autouse=True)
def instrument_mcp(tracer_provider):
    instrumenter = McpInstrumentor()
    instrumenter.instrument(tracer_provider=tracer_provider)
    yield
    instrumenter.uninstrument()


@pytest.fixture(autouse=True)
def clear_spans(span_exporter):
    yield
    span_exporter.clear()
