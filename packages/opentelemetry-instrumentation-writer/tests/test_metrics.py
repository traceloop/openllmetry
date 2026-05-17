import pytest
from opentelemetry.semconv._incubating.metrics import \
    gen_ai_metrics as GenAIMetrics
from opentelemetry.semconv_ai import Meters, SpanAttributes


@pytest.mark.vcr
def test_writer_metrics(instrument_legacy, reader, writer_client):
    writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
        stream=False,
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert data_point.attributes["gen_ai.system"] == "writer"
                        assert (
                            data_point.attributes["gen_ai.response.model"]
                            == "palmyra-x4"
                        )
                        assert data_point.attributes["llm.request.type"] == "chat"
                        assert data_point.sum > 0

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes["gen_ai.system"] == "writer"
                        assert (
                            data_point.attributes["gen_ai.response.model"]
                            == "palmyra-x4"
                        )
                        assert data_point.attributes["llm.request.type"] == "chat"
                        assert data_point.sum > 0

    assert found_token_metric
    assert found_duration_metric


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_metrics(instrument_legacy, reader, writer_client_async):
    await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
        stream=False,
    )

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert data_point.attributes["gen_ai.system"] == "writer"
                        assert (
                            data_point.attributes["gen_ai.response.model"]
                            == "palmyra-x4"
                        )
                        assert data_point.attributes["llm.request.type"] == "chat"
                        assert data_point.sum > 0

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes["gen_ai.system"] == "writer"
                        assert (
                            data_point.attributes["gen_ai.response.model"]
                            == "palmyra-x4"
                        )
                        assert data_point.attributes["llm.request.type"] == "chat"
                        assert data_point.sum > 0

    assert found_token_metric
    assert found_duration_metric


@pytest.mark.vcr
def test_writer_streaming_metrics(instrument_legacy, reader, writer_client):
    gen = writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
        stream=True,
        stream_options={"include_usage": True},
    )

    for _ in gen:
        ...

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False
    found_streaming_time_to_generate_metric = False
    found_streaming_time_to_first_token_metric = False

    total_time = 0
    time_to_first_token = 0
    time_to_generate = 0

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert data_point.attributes["gen_ai.system"] == "writer"
                        assert (
                            data_point.attributes["gen_ai.response.model"]
                            == "palmyra-x4"
                        )
                        assert data_point.attributes["llm.request.type"] == "chat"
                        assert data_point.sum > 0

                if metric.name == Meters.LLM_STREAMING_TIME_TO_GENERATE:
                    found_streaming_time_to_generate_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes["gen_ai.system"] == "writer"
                        assert (
                            data_point.attributes["gen_ai.response.model"]
                            == "palmyra-x4"
                        )
                        assert data_point.attributes["llm.request.type"] == "chat"
                        assert data_point.attributes["stream"]
                        assert data_point.sum > 0
                        time_to_generate += data_point.sum

                if metric.name == GenAIMetrics.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN:
                    found_streaming_time_to_first_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes["gen_ai.system"] == "writer"
                        assert (
                            data_point.attributes["gen_ai.response.model"]
                            == "palmyra-x4"
                        )
                        assert data_point.attributes["llm.request.type"] == "chat"
                        assert data_point.attributes["stream"]
                        assert data_point.sum > 0
                        time_to_first_token += data_point.sum

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes["gen_ai.system"] == "writer"
                        assert (
                            data_point.attributes["gen_ai.response.model"]
                            == "palmyra-x4"
                        )
                        assert data_point.attributes["llm.request.type"] == "chat"
                        assert data_point.sum > 0
                        total_time += data_point.sum

    assert found_token_metric
    assert found_duration_metric
    assert found_streaming_time_to_generate_metric
    assert found_streaming_time_to_first_token_metric
    assert total_time == time_to_first_token + time_to_generate


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_writer_async_streaming_metrics(
    instrument_legacy, reader, writer_client_async
):
    gen = await writer_client_async.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
        stream=True,
        stream_options={"include_usage": True},
    )

    async for _ in gen:
        ...

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False
    found_streaming_time_to_generate_metric = False
    found_streaming_time_to_first_token_metric = False

    total_time = 0
    time_to_first_token = 0
    time_to_generate = 0

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[SpanAttributes.LLM_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert data_point.attributes["gen_ai.system"] == "writer"
                        assert (
                            data_point.attributes["gen_ai.response.model"]
                            == "palmyra-x4"
                        )
                        assert data_point.attributes["llm.request.type"] == "chat"
                        assert data_point.sum > 0

                if metric.name == Meters.LLM_STREAMING_TIME_TO_GENERATE:
                    found_streaming_time_to_generate_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes["gen_ai.system"] == "writer"
                        assert (
                            data_point.attributes["gen_ai.response.model"]
                            == "palmyra-x4"
                        )
                        assert data_point.attributes["llm.request.type"] == "chat"
                        assert data_point.attributes["stream"]
                        assert data_point.sum > 0
                        time_to_generate += data_point.sum

                if metric.name == GenAIMetrics.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN:
                    found_streaming_time_to_first_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes["gen_ai.system"] == "writer"
                        assert (
                            data_point.attributes["gen_ai.response.model"]
                            == "palmyra-x4"
                        )
                        assert data_point.attributes["llm.request.type"] == "chat"
                        assert data_point.attributes["stream"]
                        assert data_point.sum > 0
                        time_to_first_token += data_point.sum

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes["gen_ai.system"] == "writer"
                        assert (
                            data_point.attributes["gen_ai.response.model"]
                            == "palmyra-x4"
                        )
                        assert data_point.attributes["llm.request.type"] == "chat"
                        assert data_point.sum > 0
                        total_time += data_point.sum

    assert found_token_metric
    assert found_duration_metric
    assert found_streaming_time_to_generate_metric
    assert found_streaming_time_to_first_token_metric
    assert total_time == time_to_first_token + time_to_generate
