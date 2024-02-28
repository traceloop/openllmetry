"""Unit tests configuration module."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.watsonx import WatsonxInstrumentor


pytest_plugins = []


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    try:
        # Check for required package in the env, skip test if could not found
        from ibm_watsonx_ai.foundation_models import ModelInference
        # to avoid lint error
        del ModelInference
        WatsonxInstrumentor().instrument()
    except ImportError:
        print("no supported ibm_watsonx_ai package found, Watsonx instrumentation skipped.")

    return exporter


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()


@pytest.fixture
def watson_ai_model():
    try:
        # Check for required package in the env, skip test if could not found
        from ibm_watsonx_ai.foundation_models import ModelInference
    except ImportError:
        print("no supported ibm_watsonx_ai package found, model creating skipped.")
        return None

    watsonx_ai_model = ModelInference(
        model_id="google/flan-ul2",
        project_id="c1234567-2222-2222-3333-444444444444",
        credentials={
                "apikey": "test_api_key",
                "url": "https://us-south.ml.cloud.ibm.com"
                },
    )
    return watsonx_ai_model


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization"],
        "allow_playback_repeats": True,
        "decode_compressed_response": True,
    }
