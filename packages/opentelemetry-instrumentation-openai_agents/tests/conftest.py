"""Unit tests configuration module."""

import os
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from opentelemetry.trace import set_tracer_provider


pytest_plugins = []


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    set_tracer_provider(provider)

    OpenAIAgentsInstrumentor().instrument()

    return exporter


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization", "api-key"]}


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()
