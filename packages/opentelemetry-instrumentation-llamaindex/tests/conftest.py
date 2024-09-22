"""Unit tests configuration module."""

import pytest
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.cohere import CohereInstrumentor
from opentelemetry.instrumentation.chromadb import ChromaInstrumentor
from opentelemetry.instrumentation.llamaindex import LlamaIndexInstrumentor

pytest_plugins = []


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    OpenAIInstrumentor().instrument()
    ChromaInstrumentor().instrument()
    CohereInstrumentor().instrument()
    LlamaIndexInstrumentor().instrument()

    return exporter


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()


@pytest.fixture(autouse=True)
def environment():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OPENAI_API_KEY"] = "test_api_key"
    os.environ["COHERE_API_KEY"] = "test_api_key"


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization"],
        "ignore_hosts": ["raw.githubusercontent.com"],
    }
