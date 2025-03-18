import json

import pytest

from opentelemetry.instrumentation.bedrock import PromptCaching
from opentelemetry.instrumentation.bedrock.prompt_caching import CacheSpanAttrs


def call(brt):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": "very very long system prompt",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "How do I write js?",
                        "cache_control": {
                            "type": "ephemeral"
                        }
                    },
                ]
            }
        ],
        "max_tokens": 50,
        "temperature": 0.1,
        "top_p": 0.1,
        "stop_sequences": [
            "stop"
        ],
        "top_k": 250
    }
    return brt.invoke_model(
        modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
        body=json.dumps(body),
    )


def get_metric(resource_metrics, name):
    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == name:
                    return metric
    raise Exception(f"No metric found with name {name}")


def assert_metric(reader, usage):
    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    m = get_metric(resource_metrics, PromptCaching.LLM_BEDROCK_PROMPT_CACHING)
    for data_point in m.data.data_points:
        assert data_point.attributes[CacheSpanAttrs.TYPE] in [
            "read",
            "write",
        ]
        if data_point.attributes[CacheSpanAttrs.TYPE] == "read":
            assert data_point.value == usage['cache_read_input_tokens']
        else:
            assert data_point.value == usage['cache_creation_input_tokens']


@pytest.mark.vcr
def test_prompt_cache(test_context, brt):

    _, _, reader = test_context

    response = call(brt)
    response_body = json.loads(response.get('body').read())
    # assert first prompt writes a cache
    assert response_body['usage']['cache_read_input_tokens'] == 0
    assert response_body['usage']['cache_creation_input_tokens'] > 0
    cumulative_workaround = response_body['usage']['cache_creation_input_tokens']
    assert_metric(reader, response_body['usage'])

    response = call(brt)
    response_body = json.loads(response.get('body').read())
    # assert second prompt reads from the cache
    assert response_body['usage']['cache_read_input_tokens'] > 0
    assert response_body['usage']['cache_creation_input_tokens'] == 0
    # data is stored across reads of metric data due to the cumulative behavior
    response_body['usage']['cache_creation_input_tokens'] = cumulative_workaround
    assert_metric(reader, response_body['usage'])
