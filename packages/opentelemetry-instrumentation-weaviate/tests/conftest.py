"""Unit tests configuration module."""

import os
import pytest
import weaviate
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.weaviate import WeaviateInstrumentor

pytest_plugins = []

# Check weaviate client version
_weaviate_version = tuple(int(x) for x in weaviate.__version__.split(".")[:2])
_is_v3 = _weaviate_version < (4, 0)


def pytest_collection_modifyitems(config, items):
    """Skip v4 tests if using weaviate-client v3, and vice versa."""
    skip_v4 = pytest.mark.skip(reason="weaviate-client v4 required for this test")
    skip_v3 = pytest.mark.skip(reason="weaviate-client v3 required for this test")

    for item in items:
        if "test_weaviate_instrumentation_v3" in item.nodeid and not _is_v3:
            item.add_marker(skip_v3)
        elif "test_weaviate_instrumentation.py" in item.nodeid and _is_v3:
            item.add_marker(skip_v4)


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
