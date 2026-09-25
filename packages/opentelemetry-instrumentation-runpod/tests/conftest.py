"""Unit tests configuration module."""

import pytest
import runpod
from opentelemetry.instrumentation.runpod import RunpodInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytest_plugins = []


class FakeRunPodClient:
    """
    Stands in for ``runpod.endpoint.runner.RunPodClient``.

    It speaks the same protocol as the real client - ``post(endpoint, data, timeout)``
    and ``get(endpoint, timeout)`` - and records every call instead of performing
    an HTTP request, so tests exercise the real SDK call paths with no network and
    no API key.
    """

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def post(self, endpoint, data, timeout=10):
        self.calls.append(("POST", endpoint, data, timeout))
        return self._resolve(("POST", endpoint))

    def get(self, endpoint, timeout=10):
        self.calls.append(("GET", endpoint, None, timeout))
        return self._resolve(("GET", endpoint))

    def _resolve(self, key):
        if key not in self.responses:
            raise AssertionError(f"unexpected request in test: {key}")
        response = self.responses[key]
        if isinstance(response, list):
            return response.pop(0)
        return response


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    exporter = InMemoryLogExporter()
    yield exporter


@pytest.fixture(scope="function", name="logger_provider")
def fixture_logger_provider(log_exporter):
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    return provider


PLACEHOLDER_API_KEY = "placeholder-api-key"
"""A placeholder, not a credential. The SDK refuses to build a client without one."""


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    """The SDK refuses to build a client without an API key in the environment."""
    monkeypatch.setenv("RUNPOD_API_KEY", PLACEHOLDER_API_KEY)
    monkeypatch.setattr(runpod, "api_key", PLACEHOLDER_API_KEY, raising=False)


@pytest.fixture(scope="function")
def instrument_legacy(tracer_provider):
    instrumentor = RunpodInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_content(tracer_provider, logger_provider):
    """Legacy attributes are disabled, so the content is carried by log events."""
    instrumentor = RunpodInstrumentor(use_legacy_attributes=False)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture(scope="function")
def instrument_with_no_content(tracer_provider, logger_provider, monkeypatch):
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "False")

    instrumentor = RunpodInstrumentor(use_legacy_attributes=False)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
    )

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture
def fake_client(monkeypatch):
    """
    Returns a factory that swaps the SDK's HTTP client for a
    :class:`FakeRunPodClient` on every client the SDK builds from then on.
    """
    clients = []

    def _install(responses=None):
        client = FakeRunPodClient(responses)
        clients.append(client)

        def _fake_init(self, api_key=None):  # pylint: disable=unused-argument
            self.api_key = api_key or PLACEHOLDER_API_KEY
            self.rp_session = None
            self.headers = {}
            self.endpoint_url_base = "https://api.runpod.ai/v2"
            self.post = client.post
            self.get = client.get

        monkeypatch.setattr(runpod.endpoint.runner.RunPodClient, "__init__", _fake_init)
        return client

    return _install
