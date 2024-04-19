import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    OpenAIInstrumentor(enrich_assistant=True, enrich_token_usage=True).instrument()

    return exporter


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "api-key"],
        "ignore_hosts": ["openaipublic.blob.core.windows.net"],
    }
