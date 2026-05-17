"""Unit tests configuration module."""

import os

import pytest
from opentelemetry import metrics
from opentelemetry.instrumentation.watsonx import WatsonxInstrumentor
from opentelemetry.instrumentation.watsonx.utils import TRACELOOP_TRACE_CONTENT
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource


@pytest.fixture(scope="function")
def metrics_test_context_legacy():
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
        print(
            "no supported ibm_watsonx_ai package found, Watsonx instrumentation skipped."
        )

    yield provider, reader


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    exporter = InMemoryLogExporter()
    yield exporter


@pytest.fixture(scope="function", name="logger_provider")
def fixture_logger_provider(log_exporter):
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    return provider


@pytest.fixture(scope="function")
def metrics_test_context_with_content(logger_provider):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "True"})

    resource = Resource.create()
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], resource=resource)

    metrics.set_meter_provider(provider)

    try:
        # Check for required package in the env, skip test if could not found
        from ibm_watsonx_ai.foundation_models import ModelInference

        # to avoid lint error
        del ModelInference
        WatsonxInstrumentor().instrument(logger_provider=logger_provider)
    except ImportError:
        print(
            "no supported ibm_watsonx_ai package found, Watsonx instrumentation skipped."
        )

    yield provider, reader

    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)


@pytest.fixture(scope="function")
def metrics_test_context_with_no_content(logger_provider):
    os.environ.update({TRACELOOP_TRACE_CONTENT: "False"})

    resource = Resource.create()
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], resource=resource)

    metrics.set_meter_provider(provider)

    try:
        # Check for required package in the env, skip test if could not found
        from ibm_watsonx_ai.foundation_models import ModelInference

        # to avoid lint error
        del ModelInference
        WatsonxInstrumentor().instrument(logger_provider=logger_provider)
    except ImportError:
        print(
            "no supported ibm_watsonx_ai package found, Watsonx instrumentation skipped."
        )

    yield provider, reader

    os.environ.pop(TRACELOOP_TRACE_CONTENT, None)


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization"],
        "allow_playback_repeats": True,
        "decode_compressed_response": True,
    }
