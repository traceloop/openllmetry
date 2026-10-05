import json

from opentelemetry.instrumentation.bedrock.streaming_wrapper import StreamingWrapper


def make_chunk(payload):
    return {"chunk": {"bytes": json.dumps(payload).encode()}}


class IterableList(list):
    pass


def test_streaming_tool_use_input_dict_accumulates_json():
    events = [
        make_chunk({
            "type": "message_start",
            "message": {"content": [], "usage": {}},
        }),
        make_chunk({
            "type": "content_block_start",
            "content_block": {
                "type": "tool_use",
                "id": "t1",
                "name": "get_weather",
                "input": {},
            },
        }),
        make_chunk({
            "type": "content_block_delta",
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"city": ',
            },
        }),
        make_chunk({
            "type": "content_block_delta",
            "delta": {
                "type": "input_json_delta",
                "partial_json": '"Paris"}',
            },
        }),
    ]

    result = {}

    def callback(body):
        result["body"] = body

    wrapper = StreamingWrapper(
        IterableList(events),
        stream_done_callback=callback,
    )

    for _ in wrapper:
        pass

    assert result["body"]["content"][0]["input"] == '{"city": "Paris"}'
