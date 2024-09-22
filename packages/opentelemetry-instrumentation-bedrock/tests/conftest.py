"""Unit tests configuration module."""

import os
import pytest
import boto3
from opentelemetry import trace
from opentelemetry.instrumentation.bedrock import BedrockInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytest_plugins = []


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    return exporter


@pytest.fixture(scope="session", autouse=True)
def instrument(exporter):
    BedrockInstrumentor(enrich_token_usage=True).instrument()

    yield

    exporter.shutdown()


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()


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


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization"]}
