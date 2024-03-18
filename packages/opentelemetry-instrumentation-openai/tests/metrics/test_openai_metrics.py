import os

import pytest
from openai import OpenAI


@pytest.fixture
def openai_client():
    return OpenAI()


@pytest.mark.vcr
def test_chat_completion_metrics(metrics_test_context, openai_client):
    provider, reader = metrics_test_context

    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative "
                        "flair."},
            {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
        ]
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

                if metric.name == 'llm.openai.chat_completions.tokens':
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes['llm.usage.token_type'] in ['completion', 'prompt']
                        assert len(data_point.attributes['server.address']) > 0
                        assert data_point.value > 0

                if metric.name == 'llm.openai.chat_completions.choices':
                    found_choice_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value >= 1
                        assert len(data_point.attributes['server.address']) > 0

                if metric.name == 'llm.openai.chat_completions.duration':
                    found_duration_metric = True
                    assert any(data_point.count > 0 for data_point in metric.data.data_points)
                    assert any(data_point.sum > 0 for data_point in metric.data.data_points)
                    assert all(len(data_point.attributes['server.address']) > 0
                               for data_point in metric.data.data_points)

    assert found_token_metric is True
    assert found_choice_metric is True
    assert found_duration_metric is True


@pytest.mark.vcr
def test_chat_completion_metrics_stream(metrics_test_context, openai_client):
    # set os env for token usage record in stream mode
    original_value = os.environ.get("TRACELOOP_STREAM_TOKEN_USAGE")
    os.environ["TRACELOOP_STREAM_TOKEN_USAGE"] = "true"

    try:
        provider, reader = metrics_test_context

        _ = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
            stream=True,
        )

        metrics_data = reader.get_metrics_data()
        resource_metrics = metrics_data.resource_metrics
        assert len(resource_metrics) > 0

        found_token_metric = None

        for rm in resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:

                    if metric.name == 'llm.openai.chat_completions.tokens':
                        found_token_metric = True
                        for data_point in metric.data.data_points:
                            assert data_point.attributes['llm.usage.token_type'] in ['completion', 'prompt']
                            assert len(data_point.attributes['server.address']) > 0
                            assert data_point.value > 0

        assert found_token_metric is True

    finally:
        # unset env
        if original_value is None:
            del os.environ["TRACELOOP_STREAM_TOKEN_USAGE"]
        else:
            os.environ["TRACELOOP_STREAM_TOKEN_USAGE"] = original_value


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
                if metric.name == 'llm.openai.embeddings.tokens':
                    found_token_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value > 0
                        assert len(data_point.attributes['server.address']) > 0

                if metric.name == 'llm.openai.embeddings.vector_size':
                    found_vector_size_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value > 0
                        assert len(data_point.attributes['server.address']) > 0

                if metric.name == 'llm.openai.embeddings.duration':
                    found_duration_metric = True
                    assert any(data_point.count > 0 for data_point in metric.data.data_points)
                    assert any(data_point.sum > 0 for data_point in metric.data.data_points)
                    assert all(len(data_point.attributes['server.address']) > 0
                               for data_point in metric.data.data_points)

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
                if metric.name == 'llm.openai.image_generations.duration':
                    found_duration_metric = True
                    assert any(data_point.count > 0 for data_point in metric.data.data_points)
                    assert any(data_point.sum > 0 for data_point in metric.data.data_points)
                    assert all(len(data_point.attributes['server.address']) > 0
                               for data_point in metric.data.data_points)

    assert found_duration_metric
