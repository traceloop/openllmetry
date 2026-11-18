"""The Responses API wrappers must skip when SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY is set.

The chat, completion, embedding and image wrappers already honour this key. The openai-agents
instrumentation sets it while a generation/response span is open, so without this check every
model call made through the Agents SDK was recorded twice.
"""

import asyncio
import itertools

import httpx
from openai import AsyncOpenAI, OpenAI
from opentelemetry import context as context_api
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY

RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "status": "completed",
    "model": "gpt-4.1-nano",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Paris.", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


_ids = itertools.count()


def _transport():
    # A new response id per call: the instrumentation keeps responses by id across calls.
    return httpx.MockTransport(
        lambda request: httpx.Response(200, json={**RESPONSE, "id": f"resp_{next(_ids)}"})
    )


def _sync_client():
    return OpenAI(api_key="test-key", http_client=httpx.Client(transport=_transport()))


def _async_client():
    return AsyncOpenAI(api_key="test-key", http_client=httpx.AsyncClient(transport=_transport()))


def _suppressed():
    return context_api.attach(context_api.set_value(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, True))


def test_responses_create_records_a_span_without_the_key(
    instrument_legacy, span_exporter: InMemorySpanExporter
):
    span_exporter.clear()
    _sync_client().responses.create(model="gpt-4.1-nano", input="Capital of France?")
    assert len(span_exporter.get_finished_spans()) == 1


def test_responses_create_skips_when_suppressed(
    instrument_legacy, span_exporter: InMemorySpanExporter
):
    span_exporter.clear()
    token = _suppressed()
    try:
        response = _sync_client().responses.create(model="gpt-4.1-nano", input="Capital of France?")
    finally:
        context_api.detach(token)
    assert response.output_text == "Paris."
    assert span_exporter.get_finished_spans() == ()


def test_async_responses_create_skips_when_suppressed(
    instrument_legacy, span_exporter: InMemorySpanExporter
):
    span_exporter.clear()

    async def run():
        token = _suppressed()
        try:
            return await _async_client().responses.create(model="gpt-4.1-nano", input="Capital of France?")
        finally:
            context_api.detach(token)

    response = asyncio.run(run())
    assert response.output_text == "Paris."
    assert span_exporter.get_finished_spans() == ()
