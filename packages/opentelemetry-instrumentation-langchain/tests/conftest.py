"""Unit tests configuration module."""

import os
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor

pytest_plugins = []


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    OpenAIInstrumentor().instrument()
    LangchainInstrumentor().instrument()

    return exporter


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()


@pytest.fixture(autouse=True)
def environment():
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    os.environ["ANTHROPIC_API_KEY"] = "test"
    os.environ["AWS_ACCESS_KEY_ID"] = "test_access_key_id"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test_secret_access_key"
    os.environ["AWS_SESSION_TOKEN"] = "test_session_token"


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization", "x-api-key"]}
