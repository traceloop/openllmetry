import json

import pytest

from opentelemetry.instrumentation.bedrock.streaming_wrapper import (
    AsyncStreamingWrapper,
    StreamingWrapper,
)


def make_chunk(payload):
    return {"chunk": {"bytes": json.dumps(payload).encode()}}


class IterableList(list):
    pass


class AsyncIterableList:
    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        self.iterator = iter(self.items)
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopAsyncIteration


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


@pytest.mark.asyncio
async def test_async_streaming_tool_use_input_dict_accumulates_json():
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

    wrapper = AsyncStreamingWrapper(
        AsyncIterableList(events),
        stream_done_callback=callback,
    )

    async for _ in wrapper:
        pass

    assert result["body"]["content"][0]["input"] == '{"city": "Paris"}'
