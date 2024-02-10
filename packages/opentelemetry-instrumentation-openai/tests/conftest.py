"""Unit tests configuration module."""

import pytest
import yaml
import vcr as vcr_module
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

pytest_plugins = []


class YamlSerializer:
    def deserialize(cassette_string):
        return yaml.load(cassette_string, encoding=("utf-8"))

    def serialize(cassette_dict):
        return yaml.dump(cassette_dict, encoding=("utf-8"))


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    OpenAIInstrumentor().instrument()

    return exporter


@pytest.fixture(scope="module")
def vcr():
    vcr_instance = vcr_module.VCR(
        cassette_library_dir="tests/fixtures/cassettes",
        record_mode="once",
        filter_headers=["authorization"],
    )
    vcr_instance.register_serializer("yaml-improved", YamlSerializer())
    return vcr_instance


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()


@pytest.fixture
def openai_client():
    return OpenAI()
