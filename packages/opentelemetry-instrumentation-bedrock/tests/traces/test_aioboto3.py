"""Async Bedrock instrumentation tests via aioboto3.

Mirrors the coverage of the sync test_nova.py suite — for each of the four
async methods (invoke_model, invoke_model_with_response_stream, converse,
converse_stream) there are three variants: legacy attributes, events with
content, events with no content.

# Why mocks instead of VCR cassettes

The sync tests in this package use VCR cassettes, but vcrpy's aiohttp stub
has known limitations replaying aiobotocore responses — cassettes record
correctly but replay fails with `HTTPClientError: 'NoneType' object is not
iterable` from inside aiobotocore/httpsession.py. See:
    - https://github.com/kevin1024/vcrpy/issues/927  (aiohttp streaming bodies)
    - https://github.com/kevin1024/vcrpy/issues/339  (aiohttp integration)

So these tests bypass the HTTP layer entirely by patching `_make_api_call`
on the live aioboto3 client. This still exercises every line of our
instrumentation wrappers (factory hook, async method wrappers, streaming
body wrapper, span attribute setters).
"""

import json

import aioboto3
import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiOperationNameValues,
    GenAiSystemValues,
)


MODEL_ID = "amazon.nova-lite-v1:0"
SYSTEM_LIST = [{"text": "tell me a very two sentence story."}]
MESSAGE_LIST = [{"role": "user", "content": [{"text": "A camping trip"}]}]
INF_PARAMS = {"maxTokens": 500, "topP": 0.9, "topK": 20, "temperature": 0.7}
CONVERSE_INF = {"maxTokens": 300, "topP": 0.1, "temperature": 0.3}


@pytest.fixture
def async_session():
    return aioboto3.Session(
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )


# ─────────────────────────── Mock helpers ───────────────────────────


def _fake_converse_response():
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Under a canopy of stars, the campers told stories. A bear listened."}],
            }
        },
        "usage": {"inputTokens": 11, "outputTokens": 16, "totalTokens": 27},
        "stopReason": "end_turn",
        "ResponseMetadata": {"HTTPStatusCode": 200, "HTTPHeaders": {}},
    }


def _fake_invoke_model_response():
    body_dict = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Under a canopy of stars, the campers told stories. A bear listened."}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 11, "outputTokens": 16, "totalTokens": 27},
    }
    return {
        "body": _AsyncReadableBody(json.dumps(body_dict).encode()),
        "ResponseMetadata": {"HTTPStatusCode": 200, "HTTPHeaders": {}},
        "contentType": "application/json",
    }


class _AsyncReadableBody:
    """Mimics aiobotocore's streaming body: `await body.read()` returns bytes."""

    def __init__(self, raw: bytes):
        self._raw = raw

    async def read(self, amt=None):
        if amt is None:
            out, self._raw = self._raw, b""
            return out
        out, self._raw = self._raw[:amt], self._raw[amt:]
        return out


class _AsyncEventStream:
    """Mimics aiobotocore's AioEventStream — an async-iterable object (not a
    raw async_generator), so the instrumentation can monkey-patch attributes
    on it. Also exposes _parse_event so the converse_stream instrumentation
    (which wraps that method) can hook in like it does on real aiobotocore."""

    def __init__(self, events):
        self._events = list(events)

    async def _parse_event(self, raw_event):
        # In real aiobotocore, this parses a wire-format event into a dict.
        # Our test pre-parses everything, so this is just identity.
        return raw_event

    def __aiter__(self):
        return _AsyncEventStreamIterator(self)


class _AsyncEventStreamIterator:
    def __init__(self, stream):
        self._stream = stream
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._stream._events):
            raise StopAsyncIteration
        raw = self._stream._events[self._index]
        self._index += 1
        # Route through _parse_event so the instrumentation's wrapper of
        # _parse_event observes every event (matches real aiobotocore behavior).
        return await self._stream._parse_event(raw)


def _fake_converse_stream_response():
    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Under a canopy of stars, "}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "the campers told stories."}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {"inputTokens": 11, "outputTokens": 16, "totalTokens": 27},
                "metrics": {"latencyMs": 500},
            }
        },
    ]
    return {
        "stream": _AsyncEventStream(events),
        "ResponseMetadata": {"HTTPStatusCode": 200, "HTTPHeaders": {}},
    }


def _fake_invoke_stream_response():
    """invoke_model_with_response_stream returns dict with a 'body' that yields
    `{"chunk": {"bytes": <encoded JSON>}}` events."""
    chunks = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Under a canopy of stars, "}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "the campers told stories."}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {"inputTokens": 11, "outputTokens": 16, "totalTokens": 27},
                "metrics": {"latencyMs": 500},
            }
        },
    ]
    events = [{"chunk": {"bytes": json.dumps(c).encode()}} for c in chunks]
    return {
        "body": _AsyncEventStream(events),
        "ResponseMetadata": {"HTTPStatusCode": 200, "HTTPHeaders": {}},
    }


async def _patch_make_api_call(session_fixture, response_factory):
    """Helper: opens an async bedrock client and replaces _make_api_call."""
    client_ctx = session_fixture.client("bedrock-runtime")
    client = await client_ctx.__aenter__()

    async def fake_call(operation_name, api_params):
        return response_factory()

    client._make_api_call = fake_call
    return client, client_ctx


# ─────────────────────────── converse ───────────────────────────


async def test_aioboto3_converse_legacy(
    instrument_legacy, async_session, span_exporter, log_exporter
):
    client, ctx = await _patch_make_api_call(async_session, _fake_converse_response)
    try:
        await client.converse(
            modelId=MODEL_ID,
            messages=MESSAGE_LIST,
            system=SYSTEM_LIST,
            inferenceConfig=CONVERSE_INF,
        )
    finally:
        await ctx.__aexit__(None, None, None)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "chat nova-lite-v1:0"

    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == GenAiSystemValues.AWS_BEDROCK.value
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == GenAiOperationNameValues.CHAT.value

    system_instructions = json.loads(span.attributes[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS])
    assert system_instructions[0]["content"] == SYSTEM_LIST[0]["text"]

    input_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["content"] == "A camping trip"

    output_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"

    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 300
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.3
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.1
    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, 0) > 0
    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, 0) > 0

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0


async def test_aioboto3_converse_with_events_with_content(
    instrument_with_content, async_session, span_exporter, log_exporter
):
    client, ctx = await _patch_make_api_call(async_session, _fake_converse_response)
    try:
        await client.converse(
            modelId=MODEL_ID,
            messages=MESSAGE_LIST,
            system=SYSTEM_LIST,
            inferenceConfig=CONVERSE_INF,
        )
    finally:
        await ctx.__aexit__(None, None, None)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    logs = log_exporter.get_finished_logs()
    assert len(logs) >= 2


async def test_aioboto3_converse_with_events_with_no_content(
    instrument_with_no_content, async_session, span_exporter, log_exporter
):
    client, ctx = await _patch_make_api_call(async_session, _fake_converse_response)
    try:
        await client.converse(
            modelId=MODEL_ID,
            messages=MESSAGE_LIST,
            system=SYSTEM_LIST,
            inferenceConfig=CONVERSE_INF,
        )
    finally:
        await ctx.__aexit__(None, None, None)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    logs = log_exporter.get_finished_logs()
    assert len(logs) >= 2, "events should still be emitted with content disabled"
    for log in logs:
        if log.log_record.body:
            body = dict(log.log_record.body)
            assert body == {} or "content" not in body


# ───────────────────── invoke_model (mock-based) ─────────────────────


async def test_aioboto3_invoke_model_legacy(
    instrument_legacy, async_session, span_exporter, log_exporter
):
    request_body = {
        "schemaVersion": "messages-v1",
        "messages": MESSAGE_LIST,
        "system": SYSTEM_LIST,
        "inferenceConfig": INF_PARAMS,
    }

    client, ctx = await _patch_make_api_call(async_session, _fake_invoke_model_response)
    try:
        response = await client.invoke_model(
            body=json.dumps(request_body),
            modelId=MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )
        await response["body"].read()
    finally:
        await ctx.__aexit__(None, None, None)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "chat nova-lite-v1:0"

    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == GenAiSystemValues.AWS_BEDROCK.value
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == GenAiOperationNameValues.CHAT.value

    input_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"] == [{"type": "text", "content": "A camping trip"}]

    output_messages = json.loads(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    assert len(output_messages) == 1
    assert output_messages[0]["parts"][0]["content"].startswith("Under a canopy of stars")

    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 500
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.7
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.9

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0


async def test_aioboto3_invoke_model_with_events_with_content(
    instrument_with_content, async_session, span_exporter, log_exporter
):
    request_body = {
        "schemaVersion": "messages-v1",
        "messages": MESSAGE_LIST,
        "system": SYSTEM_LIST,
        "inferenceConfig": INF_PARAMS,
    }

    client, ctx = await _patch_make_api_call(async_session, _fake_invoke_model_response)
    try:
        response = await client.invoke_model(
            body=json.dumps(request_body),
            modelId=MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )
        await response["body"].read()
    finally:
        await ctx.__aexit__(None, None, None)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    logs = log_exporter.get_finished_logs()
    assert len(logs) >= 2


async def test_aioboto3_invoke_model_with_events_with_no_content(
    instrument_with_no_content, async_session, span_exporter, log_exporter
):
    request_body = {
        "schemaVersion": "messages-v1",
        "messages": MESSAGE_LIST,
        "system": SYSTEM_LIST,
        "inferenceConfig": INF_PARAMS,
    }

    client, ctx = await _patch_make_api_call(async_session, _fake_invoke_model_response)
    try:
        response = await client.invoke_model(
            body=json.dumps(request_body),
            modelId=MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )
        await response["body"].read()
    finally:
        await ctx.__aexit__(None, None, None)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    logs = log_exporter.get_finished_logs()
    assert len(logs) >= 2, "events should still be emitted with content disabled"
    for log in logs:
        if log.log_record.body:
            body = dict(log.log_record.body)
            assert body == {} or "content" not in body


# ────────── invoke_model_with_response_stream (mock-based) ──────────


async def test_aioboto3_invoke_stream_legacy(
    instrument_legacy, async_session, span_exporter, log_exporter
):
    request_body = {
        "schemaVersion": "messages-v1",
        "messages": MESSAGE_LIST,
        "system": SYSTEM_LIST,
        "inferenceConfig": INF_PARAMS,
    }

    chunks = []
    client, ctx = await _patch_make_api_call(async_session, _fake_invoke_stream_response)
    try:
        response = await client.invoke_model_with_response_stream(
            body=json.dumps(request_body),
            modelId=MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )
        async for event in response["body"]:
            payload = json.loads(event["chunk"]["bytes"].decode())
            if "contentBlockDelta" in payload:
                chunks.append(payload["contentBlockDelta"]["delta"].get("text", ""))
    finally:
        await ctx.__aexit__(None, None, None)

    assert "".join(chunks) == "Under a canopy of stars, the campers told stories."

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "chat nova-lite-v1:0"

    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == GenAiSystemValues.AWS_BEDROCK.value
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == GenAiOperationNameValues.CHAT.value
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0


async def test_aioboto3_invoke_stream_with_events_with_content(
    instrument_with_content, async_session, span_exporter, log_exporter
):
    request_body = {
        "schemaVersion": "messages-v1",
        "messages": MESSAGE_LIST,
        "system": SYSTEM_LIST,
        "inferenceConfig": INF_PARAMS,
    }

    client, ctx = await _patch_make_api_call(async_session, _fake_invoke_stream_response)
    try:
        response = await client.invoke_model_with_response_stream(
            body=json.dumps(request_body),
            modelId=MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )
        async for _event in response["body"]:
            pass
    finally:
        await ctx.__aexit__(None, None, None)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    logs = log_exporter.get_finished_logs()
    assert len(logs) >= 2


async def test_aioboto3_invoke_stream_with_events_with_no_content(
    instrument_with_no_content, async_session, span_exporter, log_exporter
):
    request_body = {
        "schemaVersion": "messages-v1",
        "messages": MESSAGE_LIST,
        "system": SYSTEM_LIST,
        "inferenceConfig": INF_PARAMS,
    }

    client, ctx = await _patch_make_api_call(async_session, _fake_invoke_stream_response)
    try:
        response = await client.invoke_model_with_response_stream(
            body=json.dumps(request_body),
            modelId=MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )
        async for _event in response["body"]:
            pass
    finally:
        await ctx.__aexit__(None, None, None)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    logs = log_exporter.get_finished_logs()
    assert len(logs) >= 2, "events should still be emitted with content disabled"
    for log in logs:
        if log.log_record.body:
            body = dict(log.log_record.body)
            assert body == {} or "content" not in body


# ─────────────────── converse_stream (mock-based) ───────────────────


async def test_aioboto3_converse_stream_legacy(
    instrument_legacy, async_session, span_exporter, log_exporter
):
    chunks = []
    client, ctx = await _patch_make_api_call(async_session, _fake_converse_stream_response)
    try:
        response = await client.converse_stream(
            modelId=MODEL_ID,
            messages=MESSAGE_LIST,
            system=SYSTEM_LIST,
            inferenceConfig=CONVERSE_INF,
        )
        async for event in response["stream"]:
            if "contentBlockDelta" in event:
                chunks.append(event["contentBlockDelta"]["delta"].get("text", ""))
    finally:
        await ctx.__aexit__(None, None, None)

    assert "".join(chunks) == "Under a canopy of stars, the campers told stories."

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "chat nova-lite-v1:0"

    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == GenAiSystemValues.AWS_BEDROCK.value
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == GenAiOperationNameValues.CHAT.value
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "nova-lite-v1:0"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0


async def test_aioboto3_converse_stream_with_events_with_content(
    instrument_with_content, async_session, span_exporter, log_exporter
):
    client, ctx = await _patch_make_api_call(async_session, _fake_converse_stream_response)
    try:
        response = await client.converse_stream(
            modelId=MODEL_ID,
            messages=MESSAGE_LIST,
            system=SYSTEM_LIST,
            inferenceConfig=CONVERSE_INF,
        )
        async for _event in response["stream"]:
            pass
    finally:
        await ctx.__aexit__(None, None, None)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    logs = log_exporter.get_finished_logs()
    assert len(logs) >= 2


async def test_aioboto3_converse_stream_with_events_with_no_content(
    instrument_with_no_content, async_session, span_exporter, log_exporter
):
    client, ctx = await _patch_make_api_call(async_session, _fake_converse_stream_response)
    try:
        response = await client.converse_stream(
            modelId=MODEL_ID,
            messages=MESSAGE_LIST,
            system=SYSTEM_LIST,
            inferenceConfig=CONVERSE_INF,
        )
        async for _event in response["stream"]:
            pass
    finally:
        await ctx.__aexit__(None, None, None)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    logs = log_exporter.get_finished_logs()
    assert len(logs) >= 2, "events should still be emitted with content disabled"
    for log in logs:
        if log.log_record.body:
            body = dict(log.log_record.body)
            assert body == {} or "content" not in body
