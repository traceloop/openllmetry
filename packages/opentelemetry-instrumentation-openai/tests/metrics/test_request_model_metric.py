"""Focused test for https://github.com/traceloop/openllmetry/issues/3144.

Asserts that OpenAI meter datapoints carry gen_ai.request.model alongside
gen_ai.response.model. Uses InMemoryMetricReader and mocked responses
(no live API keys).
"""

from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    InMemoryMetricReader,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from opentelemetry.instrumentation.openai.shared import metric_shared_attributes
from opentelemetry.instrumentation.openai.shared.chat_wrappers import (
    _set_chat_metrics,
)


def _make_meter():
    reader = InMemoryMetricReader(
        {Counter: AggregationTemporality.CUMULATIVE,
         Histogram: AggregationTemporality.CUMULATIVE}
    )
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("test")
    token_counter = meter.create_histogram("gen_ai.client.token.usage")
    choice_counter = meter.create_counter("gen_ai.client.generation.choices")
    duration_histogram = meter.create_histogram("gen_ai.client.operation.duration")
    return reader, token_counter, choice_counter, duration_histogram


def _attributes_by_metric(reader):
    data = reader.get_metrics_data()
    out = {}
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                out[metric.name] = list(metric.data.data_points)
    return out


def test_metric_shared_attributes_includes_request_model():
    attrs = metric_shared_attributes(
        response_model="gpt-4o-2024-08-06",
        operation="chat",
        server_address="https://api.openai.com/v1/",
        is_streaming=True,
        request_model="gpt-4o",
    )
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o"
    assert attrs[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == "gpt-4o-2024-08-06"
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"


def test_chat_metrics_datapoint_has_request_model():
    reader, token_counter, choice_counter, duration_histogram = _make_meter()

    response_dict = {
        "model": "gpt-4o-2024-08-06",
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        "choices": [{"finish_reason": "stop"}],
    }
    _set_chat_metrics(
        instance=None,
        token_counter=token_counter,
        choice_counter=choice_counter,
        duration_histogram=duration_histogram,
        response_dict=response_dict,
        duration=0.5,
        is_streaming=True,
        request_model="gpt-4o",
    )

    metrics = _attributes_by_metric(reader)
    assert metrics, "expected metrics to be recorded"
    found_request_model = False
    for name, data_points in metrics.items():
        for dp in data_points:
            # every chat datapoint built via the shared helper must carry both
            assert dp.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL) == (
                "gpt-4o-2024-08-06"
            ), f"metric {name} datapoint missing response.model: {dict(dp.attributes)}"
            assert dp.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "gpt-4o", (
                f"metric {name} datapoint missing request.model: {dict(dp.attributes)}"
            )
            found_request_model = True
    assert found_request_model
