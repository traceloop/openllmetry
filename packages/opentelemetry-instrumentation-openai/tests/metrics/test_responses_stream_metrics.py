from pathlib import Path

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import Meters

# The async test replays the same recorded call, as the stream is identical
_CASSETTE = str(
    Path(__file__).parent
    / "cassettes"
    / "test_responses_stream_metrics"
    / "test_responses_stream_closed_after_completed_records_tokens.yaml"
)


def _collect_points(reader):
    data = reader.get_metrics_data()
    points = {}
    for rm in data.resource_metrics if data else []:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                points.setdefault(metric.name, []).extend(metric.data.data_points)
    return points


def _assert_token_points(points):
    durations = points.get(Meters.LLM_OPERATION_DURATION, [])
    assert [p.count for p in durations] == [1]
    assert ERROR_TYPE not in durations[0].attributes
    tokens = points.get(Meters.LLM_TOKEN_USAGE, [])
    assert {p.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE]: p.sum for p in tokens} == {"input": 12, "output": 5}
    for point in tokens:
        assert point.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
        assert point.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "openai"
        assert point.attributes["stream"] is True
        assert point.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-6-luna"


@pytest.mark.vcr
def test_responses_stream_closed_after_completed_records_tokens(
    instrument_legacy, reader, openai_client
):
    _collect_points(reader)

    stream = openai_client.responses.create(
        model="gpt-6-luna",
        input="Say hello in one word.",
        max_output_tokens=256,
        stream=True,
    )
    for event in stream:
        if event.type == "response.completed":
            break
    stream.close()

    _assert_token_points(_collect_points(reader))


@pytest.mark.vcr(_CASSETTE)
@pytest.mark.asyncio
async def test_responses_stream_closed_after_completed_records_tokens_async(
    instrument_legacy, reader, async_openai_client
):
    _collect_points(reader)

    stream = await async_openai_client.responses.create(
        model="gpt-6-luna",
        input="Say hello in one word.",
        max_output_tokens=256,
        stream=True,
    )
    async for event in stream:
        if event.type == "response.completed":
            break
    await stream.aclose()

    _assert_token_points(_collect_points(reader))
