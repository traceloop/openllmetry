import pytest
from unittest.mock import patch
from openai import OpenAI
from traceloop.sdk.decorators import workflow


@pytest.fixture
def openai_client():
    return OpenAI()


class TestInitSpansExporter:
    """Tests for init_spans_exporter with different URL schemes."""

    @pytest.mark.parametrize("endpoint", [
        "http://localhost:4318",
        "https://localhost:4318",
        "HTTP://localhost:4318",
    ])
    def test_http_schemes(self, endpoint):
        from traceloop.sdk.tracing.tracing import init_spans_exporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        assert isinstance(init_spans_exporter(endpoint, {}), OTLPSpanExporter)

    @pytest.mark.parametrize("endpoint,expected_endpoint", [
        ("http://localhost:4318", "http://localhost:4318/v1/traces"),
        ("http://localhost:4318/", "http://localhost:4318/v1/traces"),
        ("http://localhost:4318/v1/traces", "http://localhost:4318/v1/traces"),
        ("https://api.example.com", "https://api.example.com/v1/traces"),
        ("  http://localhost:4318  ", "http://localhost:4318/v1/traces"),
    ])
    def test_http_endpoint_construction(self, endpoint, expected_endpoint):
        from traceloop.sdk.tracing.tracing import init_spans_exporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        with patch.object(OTLPSpanExporter, "__init__", return_value=None) as mock:
            init_spans_exporter(endpoint, {})
            mock.assert_called_once_with(endpoint=expected_endpoint, headers={})

    @pytest.mark.parametrize("endpoint,expected_endpoint,insecure", [
        ("grpc://localhost:4317", "localhost:4317", True),
        ("GRPC://host:4317", "host:4317", True),
        ("grpcs://localhost:4317", "localhost:4317", False),
        ("GRPCS://host:4317", "host:4317", False),
        ("  grpc://localhost:4317  ", "localhost:4317", True),  # whitespace stripped
        ("localhost:4317", "localhost:4317", True),  # no scheme = insecure gRPC
    ])
    def test_grpc_schemes(self, endpoint, expected_endpoint, insecure):
        from traceloop.sdk.tracing.tracing import init_spans_exporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        with patch.object(OTLPSpanExporter, "__init__", return_value=None) as mock:
            init_spans_exporter(endpoint, {})
            mock.assert_called_once_with(endpoint=expected_endpoint, headers={}, insecure=insecure)


@pytest.mark.vcr
def test_resource_attributes(exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = exporter.get_finished_spans()
    open_ai_span = spans[0]
    assert open_ai_span.resource.attributes["something"] == "yes"
    assert open_ai_span.resource.attributes["service.name"] == "test"


@pytest.mark.vcr
def test_resource_includes_sdk_attributes(exporter, openai_client):
    """Test that resource attributes include OpenTelemetry SDK default attributes."""
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = exporter.get_finished_spans()
    span = spans[0]
    resource_attrs = span.resource.attributes

    assert resource_attrs["something"] == "yes"
    assert resource_attrs["service.name"] == "test"

    assert "telemetry.sdk.language" in resource_attrs
    assert "telemetry.sdk.name" in resource_attrs
    assert "telemetry.sdk.version" in resource_attrs

    # Verify the actual values
    assert resource_attrs["telemetry.sdk.language"] == "python"
    assert resource_attrs["telemetry.sdk.name"] == "opentelemetry"
    assert isinstance(resource_attrs["telemetry.sdk.version"], str)
    assert len(resource_attrs["telemetry.sdk.version"]) > 0


def test_custom_span_processor(exporter_with_custom_span_processor):
    @workflow()
    def run_workflow():
        pass

    run_workflow()

    spans = exporter_with_custom_span_processor.get_finished_spans()
    workflow_span = spans[0]
    assert workflow_span.attributes["custom_span"] == "yes"


@pytest.mark.vcr
def test_span_postprocess_callback(exporter_with_custom_span_postprocess_callback, openai_client):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    spans = exporter_with_custom_span_postprocess_callback.get_finished_spans()
    open_ai_span = spans[0]
    assert open_ai_span.attributes["gen_ai.prompt.0.content"] == "REDACTED"
    assert open_ai_span.attributes["gen_ai.completion.0.content"] == "REDACTED"


def test_instruments(exporter_with_custom_instrumentations):
    @workflow()
    def run_workflow():
        pass

    run_workflow()

    spans = exporter_with_custom_instrumentations.get_finished_spans()
    workflow_span = spans[0]
    assert workflow_span


def test_no_metrics(exporter_with_no_metrics):
    @workflow()
    def run_workflow():
        pass

    run_workflow()

    spans = exporter_with_no_metrics.get_finished_spans()
    workflow_span = spans[0]
    assert workflow_span


def test_multiple_span_processors(exporters_with_multiple_span_processors):
    """Test that multiple span processors work correctly together."""
    from traceloop.sdk.decorators import workflow, task

    @task(name="test_task")
    def test_task():
        return "task_result"

    @workflow(name="test_workflow")
    def test_workflow():
        return test_task()

    # Run the workflow to generate spans
    result = test_workflow()
    assert result == "task_result"

    exporters = exporters_with_multiple_span_processors

    # Check that all processors received spans
    default_spans = exporters["default"].get_finished_spans()
    custom_spans = exporters["custom"].get_finished_spans()
    metrics_spans = exporters["metrics"].get_finished_spans()

    # All processors should have received the spans
    assert len(default_spans) == 2, "Default processor should have received spans"
    assert len(custom_spans) == 2, "Custom processor should have received spans"
    assert len(metrics_spans) == 2, "Metrics processor should have received spans"

    # Verify that the default processor (Traceloop) added its attributes
    default_span = default_spans[0]
    # The default processor should have Traceloop-specific attributes
    assert hasattr(default_span, 'attributes')

    # Verify that custom processor added its attributes
    custom_span = custom_spans[0]
    assert custom_span.attributes.get("custom_processor") == "enabled"
    assert custom_span.attributes.get("processor_type") == "custom"

    # Verify that metrics processor added its attributes
    # Now that we fixed the double-call bug, the span_count should be correct
    workflow_spans = [s for s in metrics_spans if "workflow" in s.name]
    task_spans = [s for s in metrics_spans if "task" in s.name]
    assert len(workflow_spans) == 1
    assert len(task_spans) == 1

    # The workflow span should be processed first (span_count=1)
    # The task span should be processed second (span_count=2)
    workflow_span = workflow_spans[0]
    task_span = task_spans[0]

    assert workflow_span.attributes.get("metrics_processor") == "enabled"
    assert workflow_span.attributes.get("span_count") == 1

    assert task_span.attributes.get("metrics_processor") == "enabled"
    assert task_span.attributes.get("span_count") == 2


def test_get_default_span_processor():
    """Test that get_default_span_processor returns a valid processor."""
    from traceloop.sdk import Traceloop
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor

    # Test with batch disabled
    processor = Traceloop.get_default_span_processor(disable_batch=True)
    assert isinstance(processor, SimpleSpanProcessor)
    assert hasattr(processor, "_traceloop_processor")
    assert getattr(processor, "_traceloop_processor") is True

    # Test with batch enabled
    processor = Traceloop.get_default_span_processor(disable_batch=False)
    assert isinstance(processor, BatchSpanProcessor)
    assert hasattr(processor, "_traceloop_processor")
    assert getattr(processor, "_traceloop_processor") is True
