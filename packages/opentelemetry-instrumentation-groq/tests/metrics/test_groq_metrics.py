import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import Meters

MODEL = "llama3-8b-8192"


def _collect_metrics(reader):
    """Drain the reader once and return {metric name: metric}.

    The reader is configured with DELTA temporality, so a second
    `get_metrics_data()` call returns nothing.
    """
    metrics = {}
    metrics_data = reader.get_metrics_data()
    if metrics_data is None:
        return metrics
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                metrics[metric.name] = metric
    return metrics


def _assert_token_usage(metric):
    for data_point in metric.data.data_points:
        assert data_point.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE] in [
            "input",
            "output",
        ]
        assert data_point.sum > 0


@pytest.mark.vcr
def test_chat_metrics(instrument_legacy, reader, groq_client):
    groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    metrics = _collect_metrics(reader)

    assert Meters.LLM_TOKEN_USAGE in metrics
    assert Meters.LLM_OPERATION_DURATION in metrics
    _assert_token_usage(metrics[Meters.LLM_TOKEN_USAGE])


@pytest.mark.vcr
def test_chat_streaming_metrics(instrument_legacy, reader, groq_client):
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
        stream=True,
    )

    for _ in response:
        pass

    metrics = _collect_metrics(reader)

    assert Meters.LLM_TOKEN_USAGE in metrics, (
        "streaming calls record no token usage metric"
    )
    assert Meters.LLM_OPERATION_DURATION in metrics, (
        "streaming calls record no duration metric"
    )
    _assert_token_usage(metrics[Meters.LLM_TOKEN_USAGE])
