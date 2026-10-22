"""Test configuration module."""

import contextlib

import pytest
from opentelemetry.instrumentation.mcp import McpInstrumentor
from opentelemetry.instrumentation.mcp import instrumentation as instrumentation_module
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


@pytest.fixture(name="broken_tracer")
def fixture_broken_tracer():
    """A tracer that cannot open spans, the way a misconfigured SDK would.

    Instrumentation failures have to stay inside the instrumentation: with this
    tracer the traced call still runs, and its own result or failure is what the
    caller sees.
    """

    class _BrokenTracer:
        def start_as_current_span(self, *args, **kwargs):
            raise RuntimeError("tracer exploded")

    return _BrokenTracer()


@pytest.fixture(name="end_raises")
def fixture_end_raises(tracer_provider):
    """A tracer that opens real spans, but whose span end raises.

    The way a broken exporter or processor would: the failure has to stay in
    the instrumentation, and never replace the traced call's own failure.
    """
    tracer = tracer_provider.get_tracer(__name__)

    class _EndRaisingTracer:
        @contextlib.contextmanager
        def start_as_current_span(self, *args, **kwargs):
            with tracer.start_as_current_span(*args, **kwargs) as span:
                yield span
            raise RuntimeError("span end exploded")

    return _EndRaisingTracer()


@pytest.fixture(name="record_raises")
def fixture_record_raises(monkeypatch):
    """Make record_error raise, as if the span could not take the exception."""

    def _boom(*args, **kwargs):
        raise RuntimeError("record_error exploded")

    monkeypatch.setattr(instrumentation_module, "record_error", _boom)


# Instrument once for ordinary tracing tests. Lifecycle tests use a subprocess
# to exercise process-wide enable/disable without affecting this shared setup.
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
