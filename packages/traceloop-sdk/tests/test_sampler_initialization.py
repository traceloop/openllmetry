"""Tests for OpenTelemetry sampler initialization functionality."""

from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from traceloop.sdk import Traceloop
from traceloop.sdk.tracing.tracing import TracerWrapper


class TestSamplerInitialization:
    """Test class for sampler initialization functionality."""

    def test_init_with_rate_based_sampler(self):
        """Test initialization with TraceIdRatioBased sampler."""
        if hasattr(TracerWrapper, "instance"):
            _trace_wrapper_instance = TracerWrapper.instance
            del TracerWrapper.instance

        try:
            exporter = InMemorySpanExporter()
            sampler = TraceIdRatioBased(0.5)  # 50% sampling rate

            client = Traceloop.init(
                app_name="test-rate-based-sampler",
                sampler=sampler,
                exporter=exporter,
                disable_batch=True
            )

            assert client is None
            assert hasattr(TracerWrapper, "instance")
            assert TracerWrapper.instance is not None

        finally:
            if '_trace_wrapper_instance' in locals():
                TracerWrapper.instance = _trace_wrapper_instance
