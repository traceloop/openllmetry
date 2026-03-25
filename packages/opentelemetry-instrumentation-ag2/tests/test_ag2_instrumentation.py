import pytest
from unittest.mock import MagicMock, patch
from opentelemetry.instrumentation.ag2 import AG2Instrumentor


@pytest.fixture
def instrumentor():
    return AG2Instrumentor()


def test_instrument_calls_ag2_builtins(instrumentor, monkeypatch):
    """Verify that _instrument delegates to AG2's built-in opentelemetry instrumentation."""
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    mock_tracer_provider = MagicMock()

    with (
        patch("opentelemetry.instrumentation.ag2.instrumentation.wrap_function_wrapper") as mock_wrap,
        patch("autogen.opentelemetry.instrument_llm_wrapper") as mock_llm,
        patch("autogen.opentelemetry.instrument_agent"),
    ):
        instrumentor._instrument(tracer_provider=mock_tracer_provider)

        # LLM wrapper should be instrumented globally
        mock_llm.assert_called_once_with(
            tracer_provider=mock_tracer_provider,
            capture_messages=True,
        )

        # ConversableAgent.__init__ should be wrapped
        wrap_calls = [call.args[:2] for call in mock_wrap.call_args_list]
        assert ("autogen.agentchat.conversable_agent", "ConversableAgent.__init__") in wrap_calls


def test_uninstrument(instrumentor):
    """Verify that _uninstrument cleans up patches."""
    with patch("opentelemetry.instrumentation.utils.unwrap") as mock_unwrap:
        instrumentor._uninstrument()

        # Should attempt to unwrap ConversableAgent.__init__
        assert any(
            getattr(call.args[0], "__name__", None) == "ConversableAgent" and call.args[1] == "__init__"
            for call in mock_unwrap.call_args_list
        ), f"Expected unwrap call for ConversableAgent.__init__, got: {mock_unwrap.call_args_list}"
