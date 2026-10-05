"""Regression tests for #4426: non-ASCII content must survive into span attributes."""

import ast
import json
import pathlib
from typing import Callable

import httpx
import openai
import pytest

import opentelemetry.instrumentation.openai as openai_instrumentation
from opentelemetry.instrumentation.openai.utils import json_dumps
from opentelemetry.semconv_ai import SpanAttributes

PROMPT = "Какая погода в Бостоне сегодня?"
COMPLETION = "В Бостоне сегодня ясно 🌤"
TOOL_DESCRIPTION = "Узнать погоду"
TOOL_ARG_CITY = "Бостон"
SCHEMA_DESCRIPTION = "Прогноз погоды"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": TOOL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "required": ["location"],
                "properties": {"location": {"type": "string", "description": "Город"}},
            },
        },
    }
]

MESSAGES = [
    {"role": "user", "content": PROMPT},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "arguments": json.dumps({"location": TOOL_ARG_CITY}),
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "ясно"},
]

USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

CHAT_RESPONSE = {
    "id": "chatcmpl-non-ascii",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": COMPLETION},
        }
    ],
    "usage": USAGE,
}

COMPLETION_RESPONSE = {
    "id": "cmpl-non-ascii",
    "object": "text_completion",
    "created": 1,
    "model": "gpt-3.5-turbo-instruct",
    "choices": [
        {"index": 0, "text": COMPLETION, "finish_reason": "stop", "logprobs": None}
    ],
    "usage": USAGE,
}

EMBEDDINGS_RESPONSE = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
}


def _sse(chunks):
    body = "".join(
        f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks
    ) + "data: [DONE]\n\n"
    return body.encode("utf-8")


STREAM_BODY = _sse(
    [
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": COMPLETION},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> openai.OpenAI:
    # No network and no API key: only attribute serialization is under test.
    return openai.OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.fixture
def chat_client():
    return _client(lambda request: httpx.Response(200, json=CHAT_RESPONSE))


def _assert_readable(attr_value, *expected):
    for text in expected:
        assert text in attr_value
    assert "\\u04" not in attr_value  # no escaped Cyrillic
    json.loads(attr_value)  # still valid JSON


def test_chat_preserves_non_ascii(instrument_legacy, span_exporter, chat_client):
    chat_client.chat.completions.create(
        model="gpt-4o-mini", messages=MESSAGES, tools=TOOLS
    )

    attrs = span_exporter.get_finished_spans()[-1].attributes

    _assert_readable(attrs["gen_ai.input.messages"], PROMPT, TOOL_ARG_CITY)
    _assert_readable(attrs["gen_ai.tool.definitions"], TOOL_DESCRIPTION)
    _assert_readable(attrs["gen_ai.output.messages"], COMPLETION)


def test_streaming_chat_preserves_non_ascii(instrument_legacy, span_exporter):
    client = _client(
        lambda request: httpx.Response(
            200,
            content=STREAM_BODY,
            headers={"content-type": "text/event-stream"},
        )
    )

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT}],
        stream=True,
    )
    for _ in stream:
        pass

    attrs = span_exporter.get_finished_spans()[-1].attributes

    _assert_readable(attrs["gen_ai.input.messages"], PROMPT)
    _assert_readable(attrs["gen_ai.output.messages"], COMPLETION)


def test_response_format_schema_preserves_non_ascii(
    instrument_legacy, span_exporter, chat_client
):
    chat_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "weather",
                "schema": {"type": "object", "description": SCHEMA_DESCRIPTION},
            },
        },
    )

    attrs = span_exporter.get_finished_spans()[-1].attributes

    _assert_readable(
        attrs[SpanAttributes.GEN_AI_REQUEST_STRUCTURED_OUTPUT_SCHEMA],
        SCHEMA_DESCRIPTION,
    )


def test_completions_preserve_non_ascii(instrument_legacy, span_exporter):
    client = _client(lambda request: httpx.Response(200, json=COMPLETION_RESPONSE))

    client.completions.create(model="gpt-3.5-turbo-instruct", prompt=PROMPT)

    attrs = span_exporter.get_finished_spans()[-1].attributes

    _assert_readable(attrs["gen_ai.input.messages"], PROMPT)
    _assert_readable(attrs["gen_ai.output.messages"], COMPLETION)


def test_embeddings_preserve_non_ascii(instrument_legacy, span_exporter):
    client = _client(lambda request: httpx.Response(200, json=EMBEDDINGS_RESPONSE))

    client.embeddings.create(
        model="text-embedding-3-small", input=PROMPT, encoding_format="float"
    )

    attrs = span_exporter.get_finished_spans()[-1].attributes

    _assert_readable(attrs["gen_ai.input.messages"], PROMPT)


def test_no_bare_json_dumps_in_package():
    """Every JSON span attribute must go through utils.json_dumps (#4426)."""
    package_root = pathlib.Path(openai_instrumentation.__file__).parent
    offenders = []
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "dumps"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "json"
            ):
                offenders.append(f"{path.relative_to(package_root)}:{node.lineno}")

    # utils.json_dumps itself is the single allowed caller.
    offenders = [o for o in offenders if not o.startswith("utils.py:")]
    assert offenders == []


def test_json_dumps_falls_back_on_lone_surrogate():
    payload = {"content": "bad \ud83d text"}

    result = json_dumps(payload)

    result.encode("utf-8")  # must not raise
    assert result == json.dumps(payload)
