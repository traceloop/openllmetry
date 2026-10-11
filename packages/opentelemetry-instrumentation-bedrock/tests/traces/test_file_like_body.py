"""boto3 accepts a file-like `body` for invoke_model*; it must be traced
like a bytes body (the request body is parsed after the call, when botocore
has already consumed the stream)."""

import io
import json

import pytest
from botocore.response import StreamingBody
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiOperationNameValues,
)

MODEL_ID = "amazon.nova-lite-v1:0"
REQUEST_BODY = {
    "schemaVersion": "messages-v1",
    "messages": [{"role": "user", "content": [{"text": "A camping trip"}]}],
    "inferenceConfig": {"maxTokens": 500, "temperature": 0.7},
}
RESPONSE_BODY = {
    "output": {"message": {"role": "assistant", "content": [{"text": "Stars."}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 11, "outputTokens": 16, "totalTokens": 27},
}


class _EventStream:
    def __init__(self, events):
        self._events = events

    def __iter__(self):
        return iter(self._events)


def _streaming_body(raw):
    return StreamingBody(io.BytesIO(raw), len(raw))


def _stub_api_call(client, response):
    """Replace _make_api_call, consuming a file-like body like botocore does."""
    received = {}

    def fake_call(operation_name, api_params):
        body = api_params["body"]
        received["body"] = body.read() if hasattr(body, "read") else body
        return response()

    client._make_api_call = fake_call
    return received


@pytest.mark.parametrize("body_type", [bytes, io.BytesIO])
def test_invoke_model_file_like_body(instrument_legacy, brt, span_exporter, body_type):
    received = _stub_api_call(
        brt,
        lambda: {"body": _streaming_body(json.dumps(RESPONSE_BODY).encode()), "ResponseMetadata": {}},
    )

    response = brt.invoke_model(
        body=body_type(json.dumps(REQUEST_BODY).encode()),
        modelId=MODEL_ID,
        accept="application/json",
        contentType="application/json",
    )
    response["body"].read()

    assert json.loads(received["body"]) == REQUEST_BODY

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == GenAiOperationNameValues.CHAT.value
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 500
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 11
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 16


@pytest.mark.parametrize("body_type", [bytes, io.BytesIO])
def test_invoke_model_with_response_stream_file_like_body(
    instrument_legacy, brt, span_exporter, body_type
):
    chunks = [
        {"contentBlockDelta": {"delta": {"text": "Stars."}, "contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 11, "outputTokens": 16, "totalTokens": 27}}},
    ]
    events = [{"chunk": {"bytes": json.dumps(c).encode()}} for c in chunks]
    received = _stub_api_call(brt, lambda: {"body": _EventStream(events), "ResponseMetadata": {}})

    response = brt.invoke_model_with_response_stream(
        body=body_type(json.dumps(REQUEST_BODY).encode()),
        modelId=MODEL_ID,
        accept="application/json",
        contentType="application/json",
    )
    for _ in response["body"]:
        pass

    assert json.loads(received["body"]) == REQUEST_BODY

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == GenAiOperationNameValues.CHAT.value
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 500
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 11
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 16
