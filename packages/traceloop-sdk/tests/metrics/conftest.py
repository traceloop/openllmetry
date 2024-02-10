import pytest
from traceloop.sdk import Traceloop
from opentelemetry import metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture(scope="session")
def metrics_test_context():
    resource = Resource.create()
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], resource=resource)

    exporter = InMemorySpanExporter()

    metrics.set_meter_provider(provider)

    Traceloop.init(
        app_name="test",
        resource_attributes={"something": "yes"},
        disable_batch=True,
        exporter=exporter,
    )

    return provider, reader


@pytest.fixture(scope="session", autouse=True)
def clear_metrics_test_context(metrics_test_context):
    provider, reader = metrics_test_context

    reader.shutdown()
    provider.shutdown()
