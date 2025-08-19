import pytest
from openai import OpenAI
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.metrics import (
    gen_ai_metrics as GenAIMetrics,
)
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
                        data_point.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "openai"
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
                        assert len(data_point.attributes["server.address"]) > 0

                if metric.name == Meters.LLM_EMBEDDINGS_VECTOR_SIZE:
                    found_vector_size_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value > 0
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

    assert found_token_metric
    assert found_vector_size_metric
    assert found_duration_metric


@pytest.mark.vcr
def test_image_gen_metrics(instrument_legacy, reader, openai_client):
    openai_client.images.generate(
        model="dall-e-2",
        prompt="a white siamese cat",
        size="256x256",
        quality="standard",
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
                    assert all(
                        len(data_point.attributes["server.address"]) > 0
                        for data_point in metric.data.data_points
                    )

    assert found_duration_metric
