"""Regression tests: an interrupted LiteLLM call must still end its span.

``KeyboardInterrupt`` and ``asyncio.CancelledError`` derive from
``BaseException``, not ``Exception``. The wrappers used to catch only
``Exception``, so an interrupted call never ran ``_handle_exception``: the
span was never ended, never exported, and the (possibly billed) call silently
disappeared from the traces (issue #4597).

These tests drive the real instrumented ``litellm.completion`` /
``acompletion`` / ``embedding`` entry points with an httpx mock transport that
raises mid-request, so no API key or network is needed.
"""

import asyncio

import httpx
import litellm
import openai
import pytest
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import StatusCode

MESSAGES = [{"role": "user", "content": "hi"}]


def _sync_client(exc):
    def handler(request):
        raise exc

    return openai.OpenAI(
        api_key="x",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


class _CancelledTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        raise asyncio.CancelledError()


def _assert_interrupted_span(span_exporter, exc_name):
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.end_time is not None
    assert span.attributes[ERROR_TYPE] == exc_name
    assert span.status.status_code == StatusCode.ERROR


def test_completion_keyboard_interrupt_ends_span(instrument_legacy, span_exporter):
    with pytest.raises(KeyboardInterrupt):
        litellm.completion(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            client=_sync_client(KeyboardInterrupt()),
            api_key="x",
        )
    _assert_interrupted_span(span_exporter, "KeyboardInterrupt")


def test_embedding_keyboard_interrupt_ends_span(instrument_legacy, span_exporter):
    with pytest.raises(KeyboardInterrupt):
        litellm.embedding(
            model="openai/text-embedding-3-small",
            input=["hi"],
            client=_sync_client(KeyboardInterrupt()),
            api_key="x",
        )
    _assert_interrupted_span(span_exporter, "KeyboardInterrupt")


async def test_acompletion_cancelled_ends_span(instrument_legacy, span_exporter):
    client = openai.AsyncOpenAI(
        api_key="x",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=_CancelledTransport()),
    )
    with pytest.raises(asyncio.CancelledError):
        await litellm.acompletion(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            client=client,
            api_key="x",
        )
    _assert_interrupted_span(span_exporter, "CancelledError")
