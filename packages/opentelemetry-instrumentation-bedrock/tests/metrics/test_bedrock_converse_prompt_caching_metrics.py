import pytest
from opentelemetry.instrumentation.bedrock import PromptCaching
from opentelemetry.instrumentation.bedrock.prompt_caching import CacheSpanAttrs


def call(brt):
    return brt.converse(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": "What is the capital of the USA?",
                    }
                ],
            }
        ],
        inferenceConfig={"maxTokens": 50, "temperature": 0.1},
        additionalModelRequestFields={"cacheControl": {"type": "ephemeral"}},
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
            assert data_point.value == usage["cache_read_input_tokens"]
        else:
            assert data_point.value == usage["cache_creation_input_tokens"]


@pytest.mark.vcr
def test_prompt_cache_converse(test_context, brt):
    _, _, reader = test_context

    response = call(brt)
    # assert first prompt writes a cache
    usage = response["usage"]
    assert usage["cache_read_input_tokens"] == 0
    assert usage["cache_creation_input_tokens"] > 0
    cumulative_workaround = usage["cache_creation_input_tokens"]
    assert_metric(reader, usage)

    response = call(brt)
    # assert second prompt reads from the cache
    usage = response["usage"]
    assert usage["cache_read_input_tokens"] > 0
    assert usage["cache_creation_input_tokens"] == 0
    # data is stored across reads of metric data due to the cumulative behavior
    usage["cache_creation_input_tokens"] = cumulative_workaround
    assert_metric(reader, usage)