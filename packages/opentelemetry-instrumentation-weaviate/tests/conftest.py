"""Unit tests configuration module."""

import os
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.weaviate import WeaviateInstrumentor

pytest_plugins = []


def pytest_addoption(parser):
    parser.addoption(
        "--with_grpc",
        action="store_true",
        default=False,
        help=(
            "Run the tests only in case --with_grpc is specified on the command line."
            "For such tests a running weaviate instance is required."
        ),
    )


def pytest_runtest_setup(item):
    if "with_grpc" in item.keywords and not item.config.getoption("--with_grpc"):
        pytest.skip("need --with_grpc option to run this test")


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    OpenAIInstrumentor().instrument()
    WeaviateInstrumentor().instrument()

    return exporter


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()


@pytest.fixture(autouse=True)
def environment():
    os.environ["WEAVIATE_API_KEY"] = "api-key"
    os.environ["WEAVIATE_CLUSTER_URL"] = (
        "https://traceloop-sandbox-3azqlud3.weaviate.network"
    )
    os.environ["OPENAI_API_KEY"] = "open-api-key"


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "x-openai-api-key"],
    }
