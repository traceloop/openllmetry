import inspect
import json
import time
import httpx
import pytest
from openai import AsyncOpenAI, InternalServerError, OpenAI
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.metrics import (
    gen_ai_metrics as GenAIMetrics,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import Meters
from pydantic import BaseModel


@pytest.fixture
def openai_client():
    return OpenAI()


@pytest.mark.vcr
def test_chat_completion_metrics(instrument_legacy, reader, openai_client):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a poetic assistant, skilled in explaining complex programming concepts with "
                "creative flair.",
            },
            {
                "role": "user",
                "content": "Compose a poem that explains the concept of recursion in programming.",
            },
        ],
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_choice_metric = False
    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert len(data_point.attributes["server.address"]) > 0
                        assert data_point.sum > 0

                if metric.name == Meters.LLM_GENERATION_CHOICES:
                    found_choice_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value >= 1
                        assert len(data_point.attributes["server.address"]) > 0

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    assert all(
                        len(data_point.attributes["server.address"]) > 0
                        for data_point in metric.data.data_points
                    )

    assert found_token_metric is True
    assert found_choice_metric is True
    assert found_duration_metric is True


@pytest.mark.vcr
def test_chat_parsed_completion_metrics(instrument_legacy, reader, openai_client):
    class StructuredAnswer(BaseModel):
        poem: str
        style: str

    openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a poetic assistant, skilled in explaining complex programming concepts with "
                "creative flair.",
            },
            {
                "role": "user",
                "content": "Compose a poem that explains the concept of recursion in programming.",
            },
        ],
        response_format=StructuredAnswer,
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_choice_metric = False
    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                for data_point in metric.data.data_points:
                    model = data_point.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL)
                    if (
                        metric.name == Meters.LLM_TOKEN_USAGE
                        and model == "gpt-4o-2024-08-06"
                    ):
                        found_token_metric = True
                    elif (
                        metric.name == Meters.LLM_GENERATION_CHOICES
                        and model == "gpt-4o-2024-08-06"
                    ):
                        found_choice_metric = True
                    elif (
                        metric.name == Meters.LLM_OPERATION_DURATION
                        and model == "gpt-4o-2024-08-06"
                    ):
                        found_duration_metric = True

    assert found_token_metric
    assert found_choice_metric
    assert found_duration_metric


@pytest.mark.skip(reason="DeepSeek API connection issues")
@pytest.mark.vcr
def test_chat_streaming_metrics(instrument_legacy, reader, deepseek_client):
    # Since there isn't an official OpenAI API,
    # using a deepseek API that offers compatibility with the OpenAI standard.
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You are a poetic assistant, skilled in explaining complex programming concepts with "
                "creative flair.",
            },
            {
                "role": "user",
                "content": "Compose a poem that explains the concept of recursion in programming.",
            },
        ],
        stream=True,
    )

    for _ in response:
        pass

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_choice_metric = False
    found_duration_metric = False
    found_time_to_first_token_metric = False
    found_time_to_generate_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert data_point.sum > 0

                if metric.name == Meters.LLM_GENERATION_CHOICES:
                    found_choice_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value >= 1

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )

                if metric.name == GenAIMetrics.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN:
                    found_time_to_first_token_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )

                if metric.name == Meters.LLM_STREAMING_TIME_TO_GENERATE:
                    found_time_to_generate_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )

                for data_point in metric.data.data_points:
                    assert (
                        data_point.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME) == "openai"
                    )
                    # Add `deepseek-chat` to the list of models since it's a alternative to OpenAI API
                    assert str(
                        data_point.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
                    ) in ("gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4o-2024-08-06", "deepseek-chat")
                    assert data_point.attributes["gen_ai.operation.name"] == "chat"
                    assert data_point.attributes["server.address"] != ""

    assert found_token_metric is True
    assert found_choice_metric is True
    assert found_duration_metric is True
    assert found_time_to_first_token_metric is True
    assert found_time_to_generate_metric is True


@pytest.mark.vcr
def test_embeddings_metrics(instrument_legacy, reader, openai_client):
    openai_client.embeddings.create(
        input="Tell me a joke about opentelemetry",
        model="text-embedding-ada-002",
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_vector_size_metric = False
    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.sum > 0
                        if "server.address" in data_point.attributes:
                            assert len(data_point.attributes["server.address"]) > 0

                if metric.name == Meters.LLM_EMBEDDINGS_VECTOR_SIZE:
                    found_vector_size_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value > 0
                        if "server.address" in data_point.attributes:
                            assert len(data_point.attributes["server.address"]) > 0

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    for data_point in metric.data.data_points:
                        if "server.address" in data_point.attributes:
                            assert len(data_point.attributes["server.address"]) > 0

    assert found_token_metric
    assert found_vector_size_metric
    assert found_duration_metric


@pytest.mark.vcr
def test_image_gen_metrics(instrument_legacy, reader, openai_client):
    openai_client.images.generate(
        model="dall-e-2",
        prompt="a white siamese cat",
        size="256x256",
        n=1,
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    for data_point in metric.data.data_points:
                        if "server.address" in data_point.attributes:
                            assert len(data_point.attributes["server.address"]) > 0

    assert found_duration_metric


_RESPONSES_BODY = {
    "object": "response",
    "created_at": 0,
    "status": "completed",
    "model": "gpt-4o-mini",
    "output": [{
        "type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
        "content": [{"type": "output_text", "text": "hi", "annotations": []}],
    }],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 7, "output_tokens": 3, "total_tokens": 10,
        "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0},
    },
}


def _responses_ok(request):
    # A unique id per call, as already-completed ids are not traced again
    body = {**_RESPONSES_BODY, "id": f"resp_metrics_{time.time_ns()}"}
    if json.loads(request.content).get("stream"):
        event = {"type": "response.completed", "sequence_number": 0, "response": body}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"event: response.completed\ndata: {json.dumps(event)}\n\n",
        )
    return httpx.Response(200, json=body)


def _responses_server_error(request):
    return httpx.Response(500, json={"error": {"message": "boom"}})


class _BrokenStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __iter__(self):
        yield from ()
        raise httpx.ReadError("stream broke")

    async def __aiter__(self):
        for chunk in self:
            yield chunk


def _responses_broken_stream(request):
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_BrokenStream())


def _background_handler(response_id):
    def handler(request):
        status = "queued" if request.method == "POST" else "completed"
        return httpx.Response(200, json={**_RESPONSES_BODY, "id": response_id, "status": status})

    return handler


def _client(is_async, handler):
    transport = httpx.MockTransport(handler)
    if is_async:
        return AsyncOpenAI(api_key="test", max_retries=0, http_client=httpx.AsyncClient(transport=transport))
    return OpenAI(api_key="test", max_retries=0, http_client=httpx.Client(transport=transport))


async def _result(value):
    return await value if inspect.isawaitable(value) else value


async def _call_responses(is_async, handler, method="create", stream=False):
    client = _client(is_async, handler)
    kwargs = {"stream": True} if stream else {}
    response = await _result(getattr(client.responses, method)(model="gpt-4o-mini", input="hello", **kwargs))
    if stream and is_async:
        async for _ in response:
            pass
    elif stream:
        for _ in response:
            pass
    return client


def _assert_shared_attributes(attributes, client, stream):
    assert attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
    assert attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"
    assert attributes["server.address"] == str(client.base_url)
    assert attributes["stream"] is stream


def _collect_points(reader):
    data = reader.get_metrics_data()
    points = {}
    for rm in data.resource_metrics if data else []:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                points.setdefault(metric.name, []).extend(metric.data.data_points)
    return points


@pytest.mark.asyncio
@pytest.mark.parametrize("method,stream", [("create", False), ("create", True), ("parse", False)])
@pytest.mark.parametrize("is_async", [False, True])
async def test_responses_metrics(instrument_legacy, reader, is_async, method, stream):
    _collect_points(reader)

    client = await _call_responses(is_async, _responses_ok, method, stream)

    points = _collect_points(reader)
    durations = points.get(Meters.LLM_OPERATION_DURATION, [])
    assert [p.count for p in durations] == [1]
    assert durations[0].sum > 0
    assert durations[0].attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-4o-mini"
    _assert_shared_attributes(durations[0].attributes, client, stream)
    tokens = points.get(Meters.LLM_TOKEN_USAGE, [])
    assert {p.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE]: p.sum for p in tokens} == {"input": 7, "output": 3}
    for point in tokens:
        assert point.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-4o-mini"
        _assert_shared_attributes(point.attributes, client, stream)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler,stream,error",
    [
        (_responses_server_error, False, InternalServerError),
        (_responses_server_error, True, InternalServerError),
        (_responses_broken_stream, True, httpx.ReadError),
    ],
)
@pytest.mark.parametrize("is_async", [False, True])
async def test_responses_error_metrics(instrument_legacy, reader, is_async, handler, stream, error):
    _collect_points(reader)

    with pytest.raises(error):
        await _call_responses(is_async, handler, stream=stream)

    points = _collect_points(reader)
    durations = points.get(Meters.LLM_OPERATION_DURATION, [])
    assert [p.count for p in durations] == [1]
    attributes = durations[0].attributes
    assert attributes[ERROR_TYPE] == error.__name__
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    _assert_shared_attributes(attributes, _client(is_async, handler), stream)
    assert Meters.LLM_TOKEN_USAGE not in points


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_responses_stream_closed_early_records_duration(instrument_legacy, reader, is_async):
    _collect_points(reader)
    client = _client(is_async, _responses_ok)

    response = await _result(client.responses.create(model="gpt-4o-mini", input="hello", stream=True))
    await _result(response.aclose() if is_async else response.close())

    durations = _collect_points(reader).get(Meters.LLM_OPERATION_DURATION, [])
    assert [p.count for p in durations] == [1]
    assert ERROR_TYPE not in durations[0].attributes
    _assert_shared_attributes(durations[0].attributes, client, True)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_responses_background_retrieve_metrics(instrument_legacy, reader, is_async):
    _collect_points(reader)
    response_id = f"resp_background_{time.time_ns()}"
    client = _client(is_async, _background_handler(response_id))

    await _result(client.responses.create(model="gpt-4o-mini", input="hello", background=True))
    assert Meters.LLM_OPERATION_DURATION not in _collect_points(reader)

    await _result(client.responses.retrieve(response_id))
    points = _collect_points(reader)
    durations = points.get(Meters.LLM_OPERATION_DURATION, [])
    assert [p.count for p in durations] == [1]
    _assert_shared_attributes(durations[0].attributes, client, False)
    assert sorted(p.sum for p in points.get(Meters.LLM_TOKEN_USAGE, [])) == [3, 7]

    # Retrieving an already-completed response emits no span, so no new operation either
    await _result(client.responses.retrieve(response_id))
    assert _collect_points(reader) == {}
