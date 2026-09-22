import json

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI


def streaming_response(request):
    response = {
        "id": "resp_stream", "object": "response", "created_at": 1,
        "status": "completed", "model": "gpt-4o-mini", "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    event = {"type": "response.completed", "response": response, "sequence_number": 0}
    return httpx.Response(
        200, headers={"content-type": "text/event-stream", "x-request-id": "req_http"},
        content=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n", request=request,
    )


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.asyncio
async def test_async_response_stream(raw, instrument_legacy, span_exporter):
    async with AsyncOpenAI(http_client=httpx.AsyncClient(transport=httpx.MockTransport(streaming_response))) as client:
        resource = client.responses.with_raw_response if raw else client.responses
        result = await resource.create(model="gpt-4o-mini", input="hello", stream=True)
        if raw:
            assert result.headers["x-request-id"] == "req_http"
            stream = result.parse()
            assert result.parse() is stream
        else:
            stream = result
        events = [event async for event in stream]
        assert events[0].response.id == "resp_stream"
    spans = span_exporter.get_finished_spans()
    assert any(s.attributes.get("gen_ai.response.id") == "resp_stream" for s in spans)


@pytest.mark.parametrize("raw", [False, True])
def test_sync_response_stream(raw, instrument_legacy, span_exporter):
    with OpenAI(http_client=httpx.Client(transport=httpx.MockTransport(streaming_response))) as client:
        resource = client.responses.with_raw_response if raw else client.responses
        result = resource.create(model="gpt-4o-mini", input="hello", stream=True)
        if raw:
            assert result.headers["x-request-id"] == "req_http"
            stream = result.parse()
            assert result.parse() is stream
        else:
            stream = result
        assert list(stream)[0].response.id == "resp_stream"
    assert any(s.attributes.get("gen_ai.response.id") == "resp_stream" for s in span_exporter.get_finished_spans())
