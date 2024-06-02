import pytest
from openai import OpenAI
from opentelemetry.semconv.ai import SpanAttributes, Meters


@pytest.fixture
def openai_client():
    return OpenAI()


@pytest.mark.vcr
def test_chat_completion_metrics(metrics_test_context, openai_client):
    _, reader = metrics_test_context

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
                        assert data_point.attributes["gen_ai.token.type"] in [
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
def test_chat_streaming_metrics(metrics_test_context, openai_client):
    _, reader = metrics_test_context

    response = openai_client.chat.completions.create(
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
        stream=True,
    )

    for _ in response:
        pass

    reader.get_metrics_data()
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

                if metric.name == "gen_ai.client.token.usage":
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes["gen_ai.token.type"] in [
                            "output",
                            "input",
                        ]
                        assert data_point.sum > 0

                if metric.name == "gen_ai.client.generation.choices":
                    found_choice_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value >= 1

                if metric.name == "gen_ai.client.operation.duration":
                    found_duration_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )

                if (
                    metric.name
                    == "llm.openai.chat_completions.streaming_time_to_first_token"
                ):
                    found_time_to_first_token_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )

                if (
                    metric.name
                    == "llm.openai.chat_completions.streaming_time_to_generate"
                ):
                    found_time_to_generate_metric = True
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )

                for data_point in metric.data.data_points:
                    assert data_point.attributes.get("gen_ai.system") == "openai"
                    assert str(
                        data_point.attributes["gen_ai.response.model"]
                    ).startswith("gpt-3.5-turbo")
                    assert data_point.attributes["gen_ai.operation.name"] == "chat"
                    assert data_point.attributes["server.address"] != ""

    assert found_token_metric is True
    assert found_choice_metric is True
    assert found_duration_metric is True
    assert found_time_to_first_token_metric is True
    assert found_time_to_generate_metric is True


@pytest.mark.vcr
def test_embeddings_metrics(metrics_test_context, openai_client):
    provider, reader = metrics_test_context
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

                if metric.name == f"{SpanAttributes.LLM_OPENAI_EMBEDDINGS}.vector_size":
                    found_vector_size_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value > 0
                        assert len(data_point.attributes["server.address"]) > 0

                if metric.name == f"{SpanAttributes.LLM_OPENAI_EMBEDDINGS}.duration":
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
def test_image_gen_metrics(metrics_test_context, openai_client):
    provider, reader = metrics_test_context
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
