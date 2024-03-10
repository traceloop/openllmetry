"""Unit tests configuration module."""

import pytest
from opentelemetry import metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.instrumentation.watsonx import WatsonxInstrumentor


@pytest.fixture(scope="session")
def metrics_test_context():
    resource = Resource.create()
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], resource=resource)

    metrics.set_meter_provider(provider)

    try:
        # Check for required package in the env, skip test if could not found
        from ibm_watsonx_ai.foundation_models import ModelInference
        # to avoid lint error
        del ModelInference
        WatsonxInstrumentor().instrument()
    except ImportError:
        print("no supported ibm_watsonx_ai package found, Watsonx instrumentation skipped.")

    return provider, reader


@pytest.fixture(scope="session", autouse=True)
def clear_metrics_test_context(metrics_test_context):
    provider, reader = metrics_test_context

    reader.shutdown()
    provider.shutdown()


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization"],
        "allow_playback_repeats": True,
        "decode_compressed_response": True,
    }
