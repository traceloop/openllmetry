import pytest
from opentelemetry import metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from opentelemetry.instrumentation.openai import OpenAIInstrumentor


@pytest.fixture(scope="session")
def metrics_test_context():
    resource = Resource.create()
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], resource=resource)

    metrics.set_meter_provider(provider)

    OpenAIInstrumentor().instrument()

    return provider, reader


@pytest.fixture(scope="session", autouse=True)
def clear_metrics_test_context(metrics_test_context):
    provider, reader = metrics_test_context

    reader.shutdown()
    provider.shutdown()


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization"]}
