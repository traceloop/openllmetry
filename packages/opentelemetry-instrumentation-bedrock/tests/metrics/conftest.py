import os

import boto3
import pytest
from opentelemetry import metrics, trace
from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture(autouse=True)
def environment():
    if os.getenv("AWS_SECRET_ACCESS_KEY") is None:
        os.environ["AWS_ACCESS_KEY_ID"] = "test"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "test"


@pytest.fixture
def brt():
    return boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name="us-east-1",
    )


@pytest.fixture(scope="session")
def test_context():
    resource = Resource.create()
    reader = InMemoryMetricReader()
    metricProvider = MeterProvider(metric_readers=[reader], resource=resource)
    metrics.set_meter_provider(metricProvider)

    spanExporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(spanExporter)
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)

    return spanExporter, metricProvider, reader


@pytest.fixture(scope="session", autouse=True)
def instrument(test_context):
    instrumentor = BedrockInstrumentor(enrich_token_usage=True)
    instrumentor.instrument()

    yield

    exporter, provider, reader = test_context
    exporter.shutdown()
    reader.shutdown()
    provider.shutdown()


@pytest.fixture(autouse=True)
def clear_test_context(test_context):
    exporter, _, _ = test_context
    exporter.clear()
