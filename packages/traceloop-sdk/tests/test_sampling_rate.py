"""Tests for sampling_rate parameter functionality."""

import pytest
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry import trace
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task
from traceloop.sdk.tracing.tracing import TracerWrapper


def _reset_tracer_provider():
    """Helper function to properly reset the global tracer provider for testing."""
    # Reset the Once flag to allow setting a new tracer provider
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    trace._TRACER_PROVIDER = None


def test_sampling_rate_validation():
    """Test that sampling_rate validates input range."""
    if hasattr(TracerWrapper, "instance"):
        _trace_wrapper_instance = TracerWrapper.instance
        del TracerWrapper.instance

    _reset_tracer_provider()

    try:
        exporter = InMemorySpanExporter()

        with pytest.raises(ValueError, match="sampling_rate must be between 0.0 and 1.0"):
            Traceloop.init(
                app_name="test-invalid-high",
                sampling_rate=1.5,
                exporter=exporter,
                disable_batch=True
            )

        # Reset again for the second test
        _reset_tracer_provider()

        with pytest.raises(ValueError, match="sampling_rate must be between 0.0 and 1.0"):
            Traceloop.init(
                app_name="test-invalid-low",
                sampling_rate=-0.1,
                exporter=exporter,
                disable_batch=True
            )
    finally:
        _reset_tracer_provider()
        if '_trace_wrapper_instance' in locals():
            TracerWrapper.instance = _trace_wrapper_instance


def test_sampling_rate_and_sampler_conflict():
    """Test that providing both sampling_rate and sampler raises ValueError."""
    if hasattr(TracerWrapper, "instance"):
        _trace_wrapper_instance = TracerWrapper.instance
        del TracerWrapper.instance

    _reset_tracer_provider()

    try:
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
    finally:
        _reset_tracer_provider()
        if '_trace_wrapper_instance' in locals():
            TracerWrapper.instance = _trace_wrapper_instance


def test_sampling_rate_full_sampling():
    """Test that sampling_rate=1.0 samples all spans."""
    if hasattr(TracerWrapper, "instance"):
        _trace_wrapper_instance = TracerWrapper.instance
        del TracerWrapper.instance

    _reset_tracer_provider()

    try:
        exporter = InMemorySpanExporter()
        Traceloop.init(
            app_name="test-full-sampling",
            sampling_rate=1.0,
            exporter=exporter,
            disable_batch=True
        )

        @task(name="sample_task")
        def sample_task():
            return "task_result"

        @workflow(name="sample_workflow")
        def sample_workflow():
            return sample_task()

        # Execute multiple times to verify consistent sampling
        for _ in range(10):
            sample_workflow()

        spans = exporter.get_finished_spans()
        workflow_spans = [s for s in spans if s.name == "sample_workflow.workflow"]
        task_spans = [s for s in spans if s.name == "sample_task.task"]
        
        # With sampling_rate=1.0, all 10 executions should create spans
        assert len(workflow_spans) == 10
        assert len(task_spans) == 10

    finally:
        _reset_tracer_provider()
        if '_trace_wrapper_instance' in locals():
            TracerWrapper.instance = _trace_wrapper_instance


def test_sampling_rate_zero_sampling():
    """Test that sampling_rate=0.0 samples no spans."""
    if hasattr(TracerWrapper, "instance"):
        _trace_wrapper_instance = TracerWrapper.instance
        del TracerWrapper.instance

    _reset_tracer_provider()

    try:
        exporter = InMemorySpanExporter()
        Traceloop.init(
            app_name="test-zero-sampling",
            sampling_rate=0.0,
            exporter=exporter,
            disable_batch=True
        )

        @workflow(name="sample_workflow")
        def sample_workflow():
            return "result"

        # Execute multiple times
        for _ in range(10):
            sample_workflow()

        spans = exporter.get_finished_spans()
        workflow_spans = [s for s in spans if s.name == "sample_workflow.workflow"]
        
        # With sampling_rate=0.0, no spans should be created
        assert len(workflow_spans) == 0

    finally:
        _reset_tracer_provider()
        if '_trace_wrapper_instance' in locals():
            TracerWrapper.instance = _trace_wrapper_instance


def test_sampling_rate_partial_sampling():
    """Test that sampling_rate between 0 and 1 samples some spans probabilistically."""
    if hasattr(TracerWrapper, "instance"):
        _trace_wrapper_instance = TracerWrapper.instance
        del TracerWrapper.instance

    _reset_tracer_provider()

    try:
        exporter = InMemorySpanExporter()
        Traceloop.init(
            app_name="test-partial-sampling",
            sampling_rate=0.5,
            exporter=exporter,
            disable_batch=True
        )

        @workflow(name="sample_workflow")
        def sample_workflow():
            return "result"

        # Execute many times to test probabilistic sampling
        num_executions = 100
        for _ in range(num_executions):
            sample_workflow()

        spans = exporter.get_finished_spans()
        workflow_spans = [s for s in spans if s.name == "sample_workflow.workflow"]
        
        # With sampling_rate=0.5 and 100 executions, we expect roughly 50 spans
        # Allow for statistical variance (between 30 and 70 spans)
        assert 30 <= len(workflow_spans) <= 70, f"Expected 30-70 spans, got {len(workflow_spans)}"

    finally:
        _reset_tracer_provider()
        if '_trace_wrapper_instance' in locals():
            TracerWrapper.instance = _trace_wrapper_instance
