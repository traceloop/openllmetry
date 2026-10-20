"""Regression tests for https://github.com/traceloop/openllmetry/issues/4537.

The OpenAI instrumentation consumed one-shot iterables passed as ``messages``
or ``tools`` while recording request content, so the wrapped SDK call received
an exhausted iterator and the API got an empty request. The fix materializes
such iterables once and writes the list back into the call kwargs. All tests
use a mock HTTP transport — no network access or API key is needed.
"""

import json

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    _materialize_one_shot_iterables,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from .utils import get_input_messages, get_tool_definitions

MSGS = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "hi"},
]
TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
]

_COMPLETION_BODY = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"},
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.fixture
def instrumented():
    """Tracer + exporter + instrumentor wired together, torn down after."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    yield exporter
    instrumentor.uninstrument()


@pytest.fixture
def sync_client(instrumented):
    received = []
    client = OpenAI(
        api_key="x",
        base_url="http://mock/v1",
        http_client=httpx.Client(
            transport=httpx.MockTransport(_sync_handler(received))
        ),
    )
    return client, received


@pytest.fixture
def async_client(instrumented):
    received = []
    client = AsyncOpenAI(
        api_key="x",
        base_url="http://mock/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(_async_handler(received))
        ),
    )
    return client, received


def _sync_handler(received):
    def handler(request):
        body = json.loads(request.content)
        received.append(
            {
                "messages": len(body.get("messages", [])),
                "tools": len(body.get("tools", [])),
            }
        )
        return httpx.Response(200, json=_COMPLETION_BODY)

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
        return httpx.Response(200, json=_COMPLETION_BODY)

    return handler


def _chat_span(exporter):
    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["openai.chat"]
    return spans[0]


def test_messages_generator_is_not_consumed(sync_client, instrumented):
    client, received = sync_client
    client.chat.completions.create(model="m", messages=(m for m in MSGS))

    assert received[-1]["messages"] == 2
    assert len(get_input_messages(_chat_span(instrumented))) == 2


def test_tools_generator_is_not_consumed(sync_client, instrumented):
    client, received = sync_client
    client.chat.completions.create(
        model="m", messages=MSGS, tools=(t for t in TOOLS)
    )

    assert received[-1] == {"messages": 2, "tools": 1}
    tool_defs = get_tool_definitions(_chat_span(instrumented))
    assert len(tool_defs) == 1
    assert tool_defs[0]["name"] == "get_weather"


def test_map_objects_are_not_consumed(sync_client):
    client, received = sync_client
    client.chat.completions.create(
        model="m",
        messages=map(lambda m: dict(m), MSGS),
        tools=map(lambda t: dict(t), TOOLS),
    )

    assert received[-1] == {"messages": 2, "tools": 1}


def test_iter_results_are_not_consumed(sync_client):
    client, received = sync_client
    client.chat.completions.create(model="m", messages=iter(MSGS))

    assert received[-1]["messages"] == 2


@pytest.mark.asyncio
async def test_async_messages_generator_is_not_consumed(async_client):
    client, received = async_client
    await client.chat.completions.create(
        model="m", messages=(m for m in MSGS), tools=(t for t in TOOLS)
    )

    assert received[-1] == {"messages": 2, "tools": 1}


# Unit tests for the materialization helper itself.


def test_helper_materializes_generators():
    kwargs = {"messages": (m for m in MSGS), "tools": (t for t in TOOLS)}
    _materialize_one_shot_iterables(kwargs)
    assert kwargs["messages"] == MSGS
    assert kwargs["tools"] == TOOLS


def test_helper_materializes_map_objects():
    kwargs = {"messages": map(lambda m: m, MSGS)}
    _materialize_one_shot_iterables(kwargs)
    assert kwargs["messages"] == MSGS


def test_helper_leaves_lists_and_tuples_untouched():
    messages, tools = list(MSGS), tuple(TOOLS)
    kwargs = {"messages": messages, "tools": tools}
    _materialize_one_shot_iterables(kwargs)
    assert kwargs["messages"] is messages
    assert kwargs["tools"] is tools


def test_helper_ignores_missing_none_and_non_iterables():
    kwargs = {"tools": None, "functions": 42}
    _materialize_one_shot_iterables(kwargs)
    assert kwargs == {"tools": None, "functions": 42}
    _materialize_one_shot_iterables({})


def test_helper_ignores_strings():
    kwargs = {"messages": "not a message list"}
    _materialize_one_shot_iterables(kwargs)
    assert kwargs["messages"] == "not a message list"
