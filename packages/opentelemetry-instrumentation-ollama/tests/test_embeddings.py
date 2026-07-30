import importlib

import ollama
import pytest
from opentelemetry.instrumentation.ollama import OllamaInstrumentor
from opentelemetry.instrumentation.ollama.utils import TRACELOOP_TRACE_CONTENT
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


EMBED_RESPONSE = {"embeddings": [[0.1, 0.2, 0.3]]}


def _mock_ollama_requests(monkeypatch, response=EMBED_RESPONSE):
    client_module = importlib.import_module("ollama._client")
    calls = []

    def request(self, cls, *args, stream=False, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return response

    async def async_request(self, cls, *args, stream=False, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return response

    monkeypatch.setattr(client_module.Client, "_request", request)
    monkeypatch.setattr(client_module.AsyncClient, "_request", async_request)

    return calls


def _instrument_ollama(
    tracer_provider, meter_provider, logger_provider=None, use_legacy_attributes=True
):
    instrumentor = OllamaInstrumentor(use_legacy_attributes=use_legacy_attributes)
    instrument_kwargs = {
        "tracer_provider": tracer_provider,
        "meter_provider": meter_provider,
    }
    if logger_provider:
        instrument_kwargs["logger_provider"] = logger_provider
    instrumentor.instrument(**instrument_kwargs)
    return instrumentor


def _assert_embed_request(calls, expected_input):
    assert len(calls) == 1
    assert calls[0]["args"][1] == "/api/embed"
    assert calls[0]["kwargs"]["json"]["input"] == expected_input


def _assert_embed_span(ollama_span, prompt_content=None):
    assert ollama_span.name == "ollama.embeddings"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}")
        == "embedding"
    )
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "nomic-embed-text"
    )
    if prompt_content is not None:
        assert (
            ollama_span.attributes.get(
                f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"
            )
            == prompt_content
        )


def test_ollama_embed_legacy(
    monkeypatch, tracer_provider, meter_provider, span_exporter, log_exporter
):
    calls = _mock_ollama_requests(monkeypatch)
    instrumentor = _instrument_ollama(tracer_provider, meter_provider)

    try:
        response = ollama.Client().embed(
            model="nomic-embed-text", input="OpenTelemetry"
        )
    finally:
        instrumentor.uninstrument()

    assert response == EMBED_RESPONSE
    _assert_embed_request(calls, "OpenTelemetry")

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    _assert_embed_span(ollama_span, "OpenTelemetry")

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


def test_ollama_embed_multiple_inputs_legacy(
    monkeypatch, tracer_provider, meter_provider, span_exporter
):
    inputs = ["first text", "second text"]
    calls = _mock_ollama_requests(monkeypatch)
    instrumentor = _instrument_ollama(tracer_provider, meter_provider)

    try:
        ollama.Client().embed(model="nomic-embed-text", input=inputs)
    finally:
        instrumentor.uninstrument()

    _assert_embed_request(calls, inputs)

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.embeddings"
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}")
        == "embedding"
    )
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}")
        == "nomic-embed-text"
    )
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "first text"
    )
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.1.content")
        == "second text"
    )


@pytest.mark.asyncio
async def test_ollama_async_embed_with_events_with_content(
    monkeypatch,
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
    log_exporter,
):
    monkeypatch.setenv(TRACELOOP_TRACE_CONTENT, "True")
    calls = _mock_ollama_requests(monkeypatch)
    instrumentor = _instrument_ollama(
        tracer_provider,
        meter_provider,
        logger_provider=logger_provider,
        use_legacy_attributes=False,
    )

    try:
        response = await ollama.AsyncClient().embed(
            model="nomic-embed-text", input="OpenTelemetry"
        )
    finally:
        instrumentor.uninstrument()

    assert response == EMBED_RESPONSE
    _assert_embed_request(calls, "OpenTelemetry")

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    _assert_embed_span(ollama_span)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2
    assert_message_in_logs(
        logs[0], "gen_ai.user.message", {"content": "OpenTelemetry"}
    )
    assert_message_in_logs(
        logs[1],
        "gen_ai.choice",
        {
            "index": 0,
            "finish_reason": "unknown",
            "message": {"content": EMBED_RESPONSE["embeddings"]},
        },
    )


@pytest.mark.vcr
def test_ollama_embeddings_legacy(
    instrument_legacy, ollama_client, span_exporter, log_exporter
):
    ollama_client.embeddings(
        model="llama3", prompt="Tell me a joke about OpenTelemetry"
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.embeddings"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    )
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"
    assert (
        ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.0.content")
        == "Tell me a joke about OpenTelemetry"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_ollama_embeddings_with_events_with_content(
    instrument_with_content, ollama_client, span_exporter, log_exporter
):
    response = ollama_client.embeddings(
        model="llama3", prompt="Tell me a joke about OpenTelemetry"
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.embeddings"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    )
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": "Tell me a joke about OpenTelemetry"},
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response.get("embedding")},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_ollama_embeddings_with_events_with_no_content(
    instrument_with_no_content, ollama_client, span_exporter, log_exporter
):
    ollama_client.embeddings(
        model="llama3", prompt="Tell me a joke about OpenTelemetry"
    )

    spans = span_exporter.get_finished_spans()
    ollama_span = spans[0]
    assert ollama_span.name == "ollama.embeddings"
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_SYSTEM}") == "Ollama"
    assert (
        ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "embedding"
    )
    assert not ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}")
    assert ollama_span.attributes.get(f"{GenAIAttributes.GEN_AI_REQUEST_MODEL}") == "llama3"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: ReadableLogRecord, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "ollama"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
