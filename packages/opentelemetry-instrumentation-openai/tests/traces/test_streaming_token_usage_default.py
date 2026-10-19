"""Regression tests for #3168.

A standalone ``OpenAIInstrumentor()`` created without any explicit flags must
collect token usage from streaming responses. Two halves make this true:

1. The instrumentation requests usage on the wire: the OpenAI API only sends
   ``usage`` in streaming chunks when the request sets
   ``stream_options={"include_usage": True}``, so ``ensure_stream_usage``
   defaults it on streaming calls (without overriding explicit values).
2. Whatever usage the API returns in the stream is recorded on the span
   unconditionally (no opt-in flag on this instrumentor).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.openai.shared import chat_wrappers
from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    ChatStream,
    chat_wrapper,
)
from opentelemetry.instrumentation.openai.shared.config import Config
from opentelemetry.instrumentation.openai.utils import ensure_stream_usage
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

ENV_VAR = "TRACELOOP_STREAM_INCLUDE_USAGE"


def _stream_chunks():
    return [
        {
            "id": "chatcmpl-3168",
            "model": "gpt-4o-mini",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "Hello"}}
            ],
        },
        {
            "id": "chatcmpl-3168",
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"content": " world"}}],
        },
        {
            "id": "chatcmpl-3168",
            "model": "gpt-4o-mini",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        },
    ]


def _assert_usage_attributes(span):
    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 10
    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 5
    assert span.attributes.get(SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS) == 15


def test_streaming_usage_collected_with_default_instrumentor(
    tracer_provider, span_exporter
):
    # Standalone usage: no explicit flags of any kind.
    saved_config = {
        key: value for key, value in vars(Config).items() if not key.startswith("__")
    }
    try:
        OpenAIInstrumentor()

        tracer = tracer_provider.get_tracer(__name__)
        span = tracer.start_span("openai.chat")

        fake_stream = MagicMock()
        fake_stream.__next__.side_effect = [*_stream_chunks(), StopIteration]

        stream = ChatStream(
            span,
            fake_stream,
            request_kwargs={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        for _ in stream:
            pass

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        _assert_usage_attributes(spans[0])
    finally:
        for key, value in saved_config.items():
            setattr(Config, key, value)


def test_instrumented_streaming_request_includes_usage_by_default(
    tracer_provider, span_exporter, monkeypatch
):
    """An instrumented ``create(stream=True)`` without ``stream_options`` must
    still yield token usage: the wrapper adds the request option and records
    the usage the API then returns."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    saved_config = {
        key: value for key, value in vars(Config).items() if not key.startswith("__")
    }
    try:
        OpenAIInstrumentor()

        captured_kwargs = {}

        def fake_wrapped(*args, **kwargs):
            captured_kwargs.update(kwargs)
            fake_stream = MagicMock()
            fake_stream.__next__.side_effect = [*_stream_chunks(), StopIteration]
            return fake_stream

        tracer = tracer_provider.get_tracer(__name__)
        instance = SimpleNamespace(_client=object())
        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        with patch.object(
            chat_wrappers, "is_streaming_response", return_value=True
        ):
            wrapper = chat_wrapper(
                tracer, None, None, None, None, None, None
            )
            response = wrapper(fake_wrapped, instance, (), kwargs)
            for _ in response:
                pass

        # The outgoing request now asks the API for streaming usage ...
        assert captured_kwargs.get("stream_options") == {"include_usage": True}
        # ... and the returned usage lands on the finished span.
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        _assert_usage_attributes(spans[0])
    finally:
        for key, value in saved_config.items():
            setattr(Config, key, value)


class TestEnsureStreamUsage:
    def test_injects_include_usage_when_streaming(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        kwargs = {"model": "gpt-4o-mini", "stream": True}
        ensure_stream_usage(kwargs)
        assert kwargs["stream_options"] == {"include_usage": True}

    def test_preserves_explicit_stream_options(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        kwargs = {"stream": True, "stream_options": {"include_usage": False}}
        ensure_stream_usage(kwargs)
        assert kwargs["stream_options"] == {"include_usage": False}

    def test_defaults_include_usage_within_existing_options(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        kwargs = {"stream": True, "stream_options": {}}
        ensure_stream_usage(kwargs)
        assert kwargs["stream_options"] == {"include_usage": True}

    def test_noop_when_not_streaming(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        kwargs = {"model": "gpt-4o-mini"}
        ensure_stream_usage(kwargs)
        assert "stream_options" not in kwargs

    def test_env_opt_out(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "false")
        kwargs = {"model": "gpt-4o-mini", "stream": True}
        ensure_stream_usage(kwargs)
        assert "stream_options" not in kwargs

    def test_azure_requests_untouched(self, monkeypatch):
        # api-versions older than 2024-06-01 reject `stream_options`.
        monkeypatch.delenv(ENV_VAR, raising=False)
        azure_client = MagicMock(spec=openai.AzureOpenAI)
        instance = SimpleNamespace(_client=azure_client)
        kwargs = {"model": "gpt-4o-mini", "stream": True}
        ensure_stream_usage(kwargs, instance)
        assert "stream_options" not in kwargs

    def test_explicit_arg_style_request_unaffected(self):
        # Non-dict stream_options (e.g. a pydantic model) is left alone.
        kwargs = {"stream": True, "stream_options": object()}
        ensure_stream_usage(kwargs)
        assert isinstance(kwargs["stream_options"], object)


@pytest.mark.asyncio
async def test_instrumented_async_streaming_request_includes_usage_by_default(
    tracer_provider, span_exporter, monkeypatch
):
    """Async counterpart of the instrumented-request test above."""
    from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
        achat_wrapper,
    )

    monkeypatch.delenv(ENV_VAR, raising=False)
    saved_config = {
        key: value for key, value in vars(Config).items() if not key.startswith("__")
    }
    try:
        OpenAIInstrumentor()

        captured_kwargs = {}

        async def fake_wrapped(*args, **kwargs):
            captured_kwargs.update(kwargs)
            fake_stream = MagicMock()
            fake_stream.__anext__ = AsyncMock(
                side_effect=[*_stream_chunks(), StopAsyncIteration]
            )
            return fake_stream

        tracer = tracer_provider.get_tracer(__name__)
        instance = SimpleNamespace(_client=object())
        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        with patch.object(
            chat_wrappers, "is_streaming_response", return_value=True
        ):
            wrapper = achat_wrapper(
                tracer, None, None, None, None, None, None
            )
            response = await wrapper(fake_wrapped, instance, (), kwargs)
            async for _ in response:
                pass

        assert captured_kwargs.get("stream_options") == {"include_usage": True}
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        _assert_usage_attributes(spans[0])
    finally:
        for key, value in saved_config.items():
            setattr(Config, key, value)
