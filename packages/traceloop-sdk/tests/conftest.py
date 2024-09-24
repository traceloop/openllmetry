"""Unit tests configuration module."""

import os
import pytest
from traceloop.sdk import Traceloop
from traceloop.sdk.images.image_uploader import ImageUploader
from traceloop.sdk.instruments import Instruments
from traceloop.sdk.tracing.tracing import TracerWrapper
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytest_plugins = []


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    Traceloop.init(
        app_name="test",
        resource_attributes={"something": "yes"},
        disable_batch=True,
        exporter=exporter,
    )
    return exporter


@pytest.fixture(autouse=True)
def clear_exporter(exporter):
    exporter.clear()


@pytest.fixture(autouse=True)
def environment():
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "test_api_key"


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization"],
        "ignore_hosts": ["openaipublic.blob.core.windows.net"],
    }


@pytest.fixture
def exporter_with_custom_span_processor():
    # Clear singleton if existed
    if hasattr(TracerWrapper, "instance"):
        _trace_wrapper_instance = TracerWrapper.instance
        del TracerWrapper.instance

    class CustomSpanProcessor(SimpleSpanProcessor):
        def on_start(self, span, parent_context=None):
            span.set_attribute("custom_span", "yes")

    exporter = InMemorySpanExporter()
    Traceloop.init(
        exporter=exporter,
        processor=CustomSpanProcessor(exporter),
    )

    yield exporter

    # Restore singleton if any
    if _trace_wrapper_instance:
        TracerWrapper.instance = _trace_wrapper_instance


@pytest.fixture
def exporter_with_custom_instrumentations():
    # Clear singleton if existed
    if hasattr(TracerWrapper, "instance"):
        _trace_wrapper_instance = TracerWrapper.instance
        del TracerWrapper.instance

    exporter = InMemorySpanExporter()
    Traceloop.init(
        exporter=exporter,
        disable_batch=True,
        instruments=[i for i in Instruments],
    )

    yield exporter

    # Restore singleton if any
    if _trace_wrapper_instance:
        TracerWrapper.instance = _trace_wrapper_instance


@pytest.fixture
def exporter_with_no_metrics():
    # Clear singleton if existed
    if hasattr(TracerWrapper, "instance"):
        _trace_wrapper_instance = TracerWrapper.instance
        del TracerWrapper.instance

    os.environ["TRACELOOP_METRICS_ENABLED"] = "false"

    exporter = InMemorySpanExporter()

    Traceloop.init(
        exporter=exporter,
        disable_batch=True,
    )

    yield exporter

    # Restore singleton if any
    if _trace_wrapper_instance:
        TracerWrapper.instance = _trace_wrapper_instance
        os.environ["TRACELOOP_METRICS_ENABLED"] = "true"


@pytest.fixture
def image_uploader():
    return ImageUploader(base_url="https://example.com", api_key="test_api_key")
