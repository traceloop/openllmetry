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
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry import metrics

from agents import Agent
from agents.extensions.models.litellm_model import LitellmModel
from agents import ModelSettings

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


@pytest.fixture(autouse=True)
def environment():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "api-key"
    if not os.environ.get("WATSONX_URL"):
        os.environ["WATSONX_URL"] = "url"
    if not os.environ.get("WATSONX_PROJECT_ID"):
        os.environ["WATSONX_PROJECT_ID"] = "project-id"
    if not os.environ.get("WATSONX_API_KEY"):
        os.environ["WATSONX_API_KEY"] = "api-key"


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()


@pytest.fixture(scope="session")
def metrics_test_context():
    resource = Resource.create()
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], resource=resource)
    metrics.set_meter_provider(provider)
    OpenAIAgentsInstrumentor().instrument(meter_provider=provider)
    return provider, reader


@pytest.fixture(scope="session", autouse=True)
def clear_metrics_test_context(metrics_test_context):
    provider, reader = metrics_test_context
    reader.shutdown()
    provider.shutdown()


@pytest.fixture(scope="session")
def test_agent():
    test_agent = Agent(
        name="WatsonXAgent",
        instructions="You are a helpful assistant that answers all questions",
        model=LitellmModel(
            model="watsonx/meta-llama/llama-3-3-70b-instruct",
        ),
        model_settings=ModelSettings(
            temperature=0.3, max_tokens=1024, top_p=0.2, frequency_penalty=1.3
        ),
    )
    return test_agent


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization", "api-key"]}
