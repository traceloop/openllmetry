import time
import opentelemetry.sdk.metrics.export
import pytest
from openai import OpenAI


@pytest.fixture
def openai_client():
    return OpenAI()


def test_chat_completion_metrics(metrics_test_context, openai_client):
    provider, reader = metrics_test_context

    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
            {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
        ]
    )

    metrics_data = reader.get_metrics_data()  # type: opentelemetry.sdk.metrics.export.MetricsData
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_metric = False
    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == 'llm.openai.chat_completions.tokens':
                    found_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes['llm.response.model'] == 'gpt-3.5-turbo-0613'
                        assert data_point.attributes['llm.usage.token_type'] in ['completion', 'prompt']
                        assert data_point.value > 0

    assert found_metric is True

