import json
from unittest.mock import patch

import httpx

CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
}
BASE = {"id": "r1", "created": 0, "model": "mistral-large-latest"}
USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
MESSAGES = [{"role": "user", "content": "What is the weather in Paris?"}]


def _send(self, request, **kwargs):
    if json.loads(request.content).get("stream"):
        chunks = [
            BASE
            | {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [CALL | {"index": 0}],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            BASE
            | {
                "object": "chat.completion.chunk",
                "usage": USAGE,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": ""},
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
        body += "data: [DONE]\n\n"
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
            request=request,
        )
    return httpx.Response(
        200,
        json=BASE
        | {
            "object": "chat.completion",
            "usage": USAGE,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [CALL],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        request=request,
    )


def _assert_tool_call_attributes(span):
    prefix = "gen_ai.completion.0"
    assert span.attributes[f"{prefix}.role"] == "assistant"
    assert span.attributes[f"{prefix}.tool_calls.0.id"] == "call_1"
    assert span.attributes[f"{prefix}.tool_calls.0.name"] == "get_weather"
    assert json.loads(span.attributes[f"{prefix}.tool_calls.0.arguments"]) == {
        "city": "Paris"
    }


def _choice_event_body(log_exporter):
    logs = log_exporter.get_finished_logs()
    choices = [
        log for log in logs if log.log_record.event_name == "gen_ai.choice"
    ]
    assert len(choices) == 1
    return choices[0].log_record.body


def test_chat_tool_calls_legacy(instrument_legacy, mistralai_client, span_exporter):
    with patch.object(httpx.Client, "send", _send):
        mistralai_client.chat.complete(
            model="mistral-large-latest", messages=MESSAGES
        )

    (span,) = span_exporter.get_finished_spans()
    _assert_tool_call_attributes(span)


def test_chat_streaming_tool_calls_legacy(
    instrument_legacy, mistralai_client, span_exporter
):
    with patch.object(httpx.Client, "send", _send):
        for _ in mistralai_client.chat.stream(
            model="mistral-large-latest", messages=MESSAGES
        ):
            pass

    (span,) = span_exporter.get_finished_spans()
    _assert_tool_call_attributes(span)


def test_chat_tool_calls_with_events_with_content(
    instrument_with_content, mistralai_client, span_exporter, log_exporter
):
    with patch.object(httpx.Client, "send", _send):
        for _ in mistralai_client.chat.stream(
            model="mistral-large-latest", messages=MESSAGES
        ):
            pass

    body = _choice_event_body(log_exporter)
    assert body["finish_reason"] == "tool_calls"
    assert body["tool_calls"] == [
        {
            "id": "call_1",
            "function": {
                "function_name": "get_weather",
                "arguments": '{"city": "Paris"}',
            },
            "type": "function",
        }
    ]


def test_chat_tool_calls_with_events_with_no_content(
    instrument_with_no_content, mistralai_client, span_exporter, log_exporter
):
    with patch.object(httpx.Client, "send", _send):
        mistralai_client.chat.complete(
            model="mistral-large-latest", messages=MESSAGES
        )

    body = _choice_event_body(log_exporter)
    assert body["tool_calls"] == [
        {
            "id": "call_1",
            "function": {"function_name": "get_weather"},
            "type": "function",
        }
    ]
