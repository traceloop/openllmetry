"""Uninstrumenting the Ollama instrumentor must detach every wrapper it added."""

from unittest.mock import MagicMock, patch

import ollama
import pytest
from ollama._client import AsyncClient, Client
from opentelemetry.instrumentation.ollama import OllamaInstrumentor

CHAT_RESPONSE = {
    "model": "llama3",
    "created_at": "2024-01-01T00:00:00Z",
    "message": {"role": "assistant", "content": "hi"},
    "done": True,
    "prompt_eval_count": 3,
    "eval_count": 5,
}


@pytest.fixture
def instrumentor(tracer_provider):
    instrumentor = OllamaInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    yield instrumentor
    instrumentor.uninstrument()


def _fake_http_response():
    response = MagicMock()
    response.json.return_value = CHAT_RESPONSE
    return response


def _chat(client):
    return client.chat(
        model="llama3",
        messages=[{"role": "user", "content": "hi"}],
    )


def test_uninstrument_stops_tracing_ollama_calls(instrumentor, span_exporter):
    client = ollama.Client(host="http://localhost:11434")

    with patch("httpx.Client.send", return_value=_fake_http_response()):
        _chat(client)
        assert len(span_exporter.get_finished_spans()) == 1

        instrumentor.uninstrument()
        _chat(client)

    assert len(span_exporter.get_finished_spans()) == 1


def test_uninstrument_restores_the_wrapped_request_dispatchers(instrumentor):
    assert hasattr(Client._request, "__wrapped__")
    assert hasattr(AsyncClient._request, "__wrapped__")

    instrumentor.uninstrument()

    assert not hasattr(Client._request, "__wrapped__")
    assert not hasattr(AsyncClient._request, "__wrapped__")


def test_reinstrument_after_uninstrument_does_not_stack_spans(
    instrumentor, tracer_provider, span_exporter
):
    instrumentor.uninstrument()
    instrumentor.instrument(tracer_provider=tracer_provider)

    client = ollama.Client(host="http://localhost:11434")
    with patch("httpx.Client.send", return_value=_fake_http_response()):
        _chat(client)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "ollama.chat"
