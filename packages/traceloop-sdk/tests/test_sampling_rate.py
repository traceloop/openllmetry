"""Tests for sampling_rate parameter functionality."""

import pytest
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from traceloop.sdk import Traceloop
from traceloop.sdk.tracing.tracing import TracerWrapper


@pytest.fixture
def clean_tracer_wrapper():
    """Fixture to manage TracerWrapper global state."""
    original_instance = None
    if hasattr(TracerWrapper, "instance"):
        original_instance = TracerWrapper.instance
        del TracerWrapper.instance
    
    yield
    
    if hasattr(TracerWrapper, "instance"):
        del TracerWrapper.instance
    if original_instance is not None:
        TracerWrapper.instance = original_instance


class TestSamplingRate:
    """Test class for sampling_rate parameter functionality."""

    def test_init_with_sampling_rate(self, clean_tracer_wrapper):
        """Test initialization with sampling_rate parameter."""
        exporter = InMemorySpanExporter()

        client = Traceloop.init(
            app_name="test-sampling-rate",
            sampling_rate=0.5,
            exporter=exporter,
            disable_batch=True
        )

        assert client is None
        assert hasattr(TracerWrapper, "instance")
        assert TracerWrapper.instance is not None
        
        # Verify that a tracer provider was created with a sampler
        tracer_provider = TracerWrapper.instance._TracerWrapper__tracer_provider
        assert tracer_provider is not None
        assert tracer_provider.sampler is not None

    def test_sampling_rate_validation(self, clean_tracer_wrapper):
        """Test that sampling_rate validates input range."""
        exporter = InMemorySpanExporter()

        with pytest.raises(ValueError, match="sampling_rate must be between 0.0 and 1.0"):
            Traceloop.init(
                app_name="test-invalid-sampling-rate",
                sampling_rate=1.5,
                exporter=exporter,
                disable_batch=True
            )

        with pytest.raises(ValueError, match="sampling_rate must be between 0.0 and 1.0"):
            Traceloop.init(
                app_name="test-invalid-sampling-rate",
                sampling_rate=-0.1,
                exporter=exporter,
                disable_batch=True
            )

    def test_sampling_rate_and_sampler_conflict(self, clean_tracer_wrapper):
        """Test that providing both sampling_rate and sampler raises an error."""
        exporter = InMemorySpanExporter()
        sampler = TraceIdRatioBased(0.5)

        with pytest.raises(ValueError, match="Cannot specify both 'sampler' and 'sampling_rate'"):
            Traceloop.init(
                app_name="test-conflict",
                sampling_rate=0.5,
                sampler=sampler,
                exporter=exporter,
                disable_batch=True
            )

    def test_sampling_rate_edge_cases(self, clean_tracer_wrapper):
        """Test sampling_rate with edge case values (0.0 and 1.0)."""
        exporter = InMemorySpanExporter()

        # Test 0.0 (no sampling)
        client = Traceloop.init(
            app_name="test-sampling-zero",
            sampling_rate=0.0,
            exporter=exporter,
            disable_batch=True
        )
        assert client is None
        assert hasattr(TracerWrapper, "instance")
        
        tracer_provider = TracerWrapper.instance._TracerWrapper__tracer_provider
        assert tracer_provider.sampler is not None

        del TracerWrapper.instance

        # Test 1.0 (full sampling)
        client = Traceloop.init(
            app_name="test-sampling-one",
            sampling_rate=1.0,
            exporter=exporter,
            disable_batch=True
        )
        assert client is None
        assert hasattr(TracerWrapper, "instance")
        
        tracer_provider = TracerWrapper.instance._TracerWrapper__tracer_provider
        assert tracer_provider.sampler is not None
