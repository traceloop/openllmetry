import pytest
from unittest.mock import MagicMock, patch
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import set_tracer_provider
from opentelemetry.instrumentation.mcp import McpInstrumentor
from mcp.server.lowlevel.server import Server
from mcp.shared.session import BaseSession


@pytest.fixture
def tracer_provider_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(provider)
    yield provider, exporter
    exporter.clear()


@patch("mcp.server.lowlevel.server.Server._handle_request")
@patch("mcp.shared.session.BaseSession.send_request")
def test_instrumentation_creates_spans(
    mock_send_request, mock_handle_request, tracer_provider_and_exporter
):
    provider, exporter = tracer_provider_and_exporter

    instrumentor = McpInstrumentor()
    instrumentor.instrument(tracer_provider=provider)

    mock_handle_request.return_value = "mock_response"
    mock_send_request.return_value = "mock_response"

    mock_request = MagicMock()
    mock_request.method = "TestMethod"
    mock_request.id = "test-id"
    mock_request.params = MagicMock()
    mock_request.params.meta = MagicMock()
    mock_request.params.meta.traceparent = None

    mock_session = MagicMock()
    mock_session._init_options = {"option": "value"}

    Server("Test")._handle_request(None, mock_request, mock_session)

    mock_args = MagicMock()
    mock_args.root = MagicMock()
    mock_args.root.method = "TestMethod"
    mock_args.root.params = MagicMock()
    mock_args.root.params.meta = {}

    base_session_instance = MagicMock(spec=BaseSession)

    base_session_instance.send_request(mock_args)

    spans = exporter.get_finished_spans()

    assert len(spans) >= 0
    methods = [span.attributes.get("mcp.method.name") for span in spans]
    assert "TestMethod" in methods
