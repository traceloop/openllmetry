"""Regression test for #3168.

A standalone ``OpenAIInstrumentor()`` created without any explicit flags must
collect token usage from streaming responses. The instrumentation records
whatever usage the API returns in the stream unconditionally (there is no
opt-in flag for this on the OpenAI instrumentor), so this test drives the
real ``ChatStream`` accumulation path with a mocked stream and asserts the
usage attributes land on the finished span.
"""

from unittest.mock import MagicMock

from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.openai.shared.chat_wrappers import ChatStream
from opentelemetry.instrumentation.openai.shared.config import Config
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


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


def test_streaming_usage_collected_with_default_instrumentor(
    tracer_provider, span_exporter
):
    # Snapshot global Config: constructing the instrumentor below must not
    # leak state into other tests sharing this session.
    saved_config = {
        key: value for key, value in vars(Config).items() if not key.startswith("__")
    }
    try:
        # Standalone usage: no explicit flags of any kind.
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
        finished = spans[0]
        assert (
            finished.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
            == 10
        )
        assert (
            finished.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
            == 5
        )
        assert (
            finished.attributes.get(SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS)
            == 15
        )
    finally:
        for key, value in saved_config.items():
            setattr(Config, key, value)
