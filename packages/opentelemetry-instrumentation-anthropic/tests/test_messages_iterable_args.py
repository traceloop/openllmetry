"""Regression tests for https://github.com/traceloop/openllmetry/issues/4537.

The Anthropic instrumentation consumed one-shot iterables passed as
``messages`` or ``tools`` while recording request content, so the wrapped SDK
call received an exhausted iterator and the API got an empty request. The fix
materializes such iterables once and writes the list back into the call
kwargs. All tests use a mock HTTP transport — no network access or API key is
needed.
"""

import json

import pytest

try:
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx
from anthropic import Anthropic, AsyncAnthropic
from opentelemetry.instrumentation.anthropic import (
    AnthropicInstrumentor,
    _materialize_one_shot_iterables,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

MSGS = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
    {"role": "user", "content": "weather?"},
]
TOOLS = [{"name": "get_weather", "input_schema": {"type": "object"}}]

_MESSAGE_BODY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "m",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


def _sync_handler(received):
    def handler(request):
        body = json.loads(request.content)
        received.append(
            {
                "messages": len(body.get("messages", [])),
                "tools": len(body.get("tools", [])),
            }
        )
        return httpx.Response(200, json=_MESSAGE_BODY)

    return handler


def _async_handler(received):
    async def handler(request):
        body = json.loads(request.content)
        received.append(
            {
                "messages": len(body.get("messages", [])),
                "tools": len(body.get("tools", [])),
            }
        )
        return httpx.Response(200, json=_MESSAGE_BODY)

    return handler


@pytest.fixture
def instrumented():
    """Tracer + exporter + instrumentor wired together, torn down after."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    yield exporter
    instrumentor.uninstrument()


@pytest.fixture
def sync_client(instrumented):
    received = []
    client = Anthropic(
        api_key="x",
        base_url="http://mock",
        http_client=httpx.Client(
            transport=httpx.MockTransport(_sync_handler(received))
        ),
    )
    return client, received


@pytest.fixture
def async_client(instrumented):
    received = []
    client = AsyncAnthropic(
        api_key="x",
        base_url="http://mock",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(_async_handler(received))
        ),
    )
    return client, received


def _chat_span(exporter):
    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["anthropic.chat"]
    return spans[0]


def test_messages_generator_is_not_consumed(sync_client, instrumented):
    client, received = sync_client
    client.messages.create(
        model="m", max_tokens=10, messages=(m for m in MSGS)
    )

    assert received[-1]["messages"] == 3
    input_messages = json.loads(
        _chat_span(instrumented).attributes["gen_ai.input.messages"]
    )
    assert len(input_messages) == 3


def test_tools_generator_is_not_consumed(sync_client, instrumented):
    client, received = sync_client
    client.messages.create(
        model="m", max_tokens=10, messages=MSGS, tools=(t for t in TOOLS)
    )

    assert received[-1] == {"messages": 3, "tools": 1}
    tool_defs = json.loads(
        _chat_span(instrumented).attributes["gen_ai.tool.definitions"]
    )
    assert len(tool_defs) == 1
    assert tool_defs[0]["name"] == "get_weather"


def test_map_objects_are_not_consumed(sync_client):
    client, received = sync_client
    client.messages.create(
        model="m", max_tokens=10, messages=map(lambda m: dict(m), MSGS)
    )

    assert received[-1]["messages"] == 3


@pytest.mark.asyncio
async def test_async_generators_are_not_consumed(async_client):
    client, received = async_client
    await client.messages.create(
        model="m",
        max_tokens=10,
        messages=(m for m in MSGS),
        tools=(t for t in TOOLS),
    )

    assert received[-1] == {"messages": 3, "tools": 1}


# Unit tests for the materialization helper itself.


def test_helper_materializes_generators():
    kwargs = {"messages": (m for m in MSGS), "tools": (t for t in TOOLS)}
    _materialize_one_shot_iterables(kwargs)
    assert kwargs["messages"] == MSGS
    assert kwargs["tools"] == TOOLS


def test_helper_leaves_lists_and_tuples_untouched():
    messages, system = list(MSGS), ("be brief",)
    kwargs = {"messages": messages, "system": system}
    _materialize_one_shot_iterables(kwargs)
    assert kwargs["messages"] is messages
    assert kwargs["system"] is system


def test_helper_ignores_missing_none_and_non_iterables():
    kwargs = {"tools": None, "messages": 42}
    _materialize_one_shot_iterables(kwargs)
    assert kwargs == {"tools": None, "messages": 42}
    _materialize_one_shot_iterables({})
