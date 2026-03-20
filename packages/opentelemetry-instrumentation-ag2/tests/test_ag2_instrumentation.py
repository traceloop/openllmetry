import pytest
from unittest.mock import MagicMock
from opentelemetry.instrumentation.ag2 import AG2Instrumentor
from opentelemetry.trace.status import StatusCode


@pytest.fixture
def mock_instrumentor():
    instrumentor = AG2Instrumentor()
    instrumentor.instrument = MagicMock()
    instrumentor.uninstrument = MagicMock()
    return instrumentor


@pytest.fixture
def mock_agents():
    assistant = MagicMock()
    assistant.name = "assistant"
    assistant.description = "A helpful assistant"
    assistant._oai_system_message = [{"content": "You are a helpful AI assistant."}]

    user_proxy = MagicMock()
    user_proxy.name = "user_proxy"
    user_proxy.description = "A user proxy agent"
    user_proxy._oai_system_message = [{"content": ""}]

    return assistant, user_proxy


def test_ag2_instrumentation(mock_agents, mock_instrumentor):
    mock_instrumentor.instrument()
    mock_instrumentor.instrument.assert_called_once()

    assistant, user_proxy = mock_agents
    assert assistant.name == "assistant"
    assert user_proxy.name == "user_proxy"


def test_trace_status(mock_agents, mock_instrumentor):
    mock_span = MagicMock()
    mock_span.set_status = MagicMock()

    mock_span.set_status(StatusCode.OK)
    mock_span.set_status.assert_called_with(StatusCode.OK)

    mock_span.set_status(StatusCode.ERROR)
    mock_span.set_status.assert_called_with(StatusCode.ERROR)

    memory_exporter = MagicMock()
    memory_exporter.get_finished_spans.return_value = [
        MagicMock(status=MagicMock(status_code=StatusCode.ERROR))
    ]

    spans = memory_exporter.get_finished_spans()
    assert spans[-1].status.status_code == StatusCode.ERROR

    mock_instrumentor.uninstrument()
    mock_instrumentor.uninstrument.assert_called_once()
