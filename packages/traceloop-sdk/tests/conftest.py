"""Unit tests configuration module."""

import pytest
from traceloop.sdk import Traceloop
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


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization"]}
