import pytest
from unittest.mock import MagicMock, patch, Mock
from opentelemetry.trace import get_tracer
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import set_tracer_provider
from opentelemetry.instrumentation.mcp import McpInstrumentor 
from mcp.server.lowlevel.server import Server
from mcp.shared.session import BaseSession

class TestMcpInstrumentor:

    def setup_method(self):
        self.instrumentor = McpInstrumentor()
        self.tracer_provider = Mock()
        self.tracer = get_tracer("test_module", "1.0.0", self.tracer_provider)

    def test_instrumentation_dependencies(self):
        assert self.instrumentor.instrumentation_dependencies() == ("mcp >= 1.6.0",)

    @patch("your_module.register_post_import_hook")
    def test__instrument(self, mock_register_post_import_hook):
        self.instrumentor._instrument(tracer_provider=self.tracer_provider)
        mock_register_post_import_hook.assert_called_with(
            Mock(), "mcp.client.sse"
        )
        mock_register_post_import_hook.assert_called_with(
            Mock(), "mcp.server.sse"
        )
        mock_register_post_import_hook.assert_called_with(
            Mock(), "mcp.client.stdio"
        )
        mock_register_post_import_hook.assert_called_with(
            Mock(), "mcp.server.stdio"
        )
        mock_register_post_import_hook.assert_called_with(
            Mock(), "mcp.server.session"
        )

    @patch("your_module.wrap_function_wrapper")
    def test_uninstrument(self, mock_wrap_function_wrapper):
        self.instrumentor._uninstrument()
        mock_wrap_function_wrapper.assert_called_with(
            "mcp.client.stdio", "stdio_client"
        )
        mock_wrap_function_wrapper.assert_called_with(
            "mcp.server.stdio", "stdio_server"
        )

    @patch("your_module.asynccontextmanager")
    def test__transport_wrapper(self, mock_asynccontextmanager):
        traced_method = self.instrumentor._transport_wrapper(self.tracer)
        assert isinstance(traced_method, Mock)
        mock_asynccontextmanager.assert_called_with(
            Mock(),
            Mock(),
            Mock(),
            Mock(),
        )

    @patch("your_module.wrap_function_wrapper")
    def test__base_session_init_wrapper(self, mock_wrap_function_wrapper):
        traced_method = self.instrumentor._base_session_init_wrapper(self.tracer)
        assert isinstance(traced_method, Mock)
        mock_wrap_function_wrapper.assert_called_with(
            Mock(),
            Mock(),
            Mock(),
            Mock(),
        )

    @patch("your_module.set_span_in_context")
    @patch("your_module.TraceContextTextMapPropagator")
    def test_patch_mcp_client(self, mock_trace_context_text_map_propagator, mock_set_span_in_context):
        traced_method = self.instrumentor.patch_mcp_client(self.tracer)
        assert isinstance(traced_method, Mock)
        mock_trace_context_text_map_propagator.assert_called_with()
        mock_set_span_in_context.assert_called_with()

if __name__ == "__main__":
    pytest.main()
