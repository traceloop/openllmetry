"""Regression test for the Bedrock operation duration metric.

``gen_ai.client.operation.duration`` must measure the client call. The
recorder used to read ``metric_params.start_time``, which is set when the
``bedrock-runtime`` client is created, so every recorded duration was the
client's age instead of the call's latency (issue #4502).
"""

import time
from typing import Any

from opentelemetry.instrumentation.bedrock import MetricParams, _instrumented_converse
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv_ai import Meters

MODEL_ID = "us.openai.gpt-6-sol"

CONVERSE_RESPONSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "pong"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
    # Real responses omit this key when no guardrail is configured;
    # is_guardrail_activated() currently treats the missing key as an activation.
    "amazon-bedrock-guardrailAction": "NONE",
}


def _metric_params(meter) -> MetricParams:
    return MetricParams(
        token_histogram=meter.create_histogram(Meters.LLM_TOKEN_USAGE),
        choice_counter=meter.create_counter(Meters.LLM_GENERATION_CHOICES),
        duration_histogram=meter.create_histogram(Meters.LLM_OPERATION_DURATION),
        exception_counter=meter.create_counter("gen_ai.client.operation.exception"),
        guardrail_activation=meter.create_counter("gen_ai.bedrock.guardrail.activation"),
        guardrail_latency_histogram=meter.create_histogram("gen_ai.bedrock.guardrail.latency"),
        guardrail_coverage=meter.create_counter("gen_ai.bedrock.guardrail.coverage"),
        guardrail_sensitive_info=meter.create_counter("gen_ai.bedrock.guardrail.sensitive_info"),
        guardrail_topic=meter.create_counter("gen_ai.bedrock.guardrail.topic"),
        guardrail_content=meter.create_counter("gen_ai.bedrock.guardrail.content"),
        guardrail_words=meter.create_counter("gen_ai.bedrock.guardrail.words"),
        prompt_caching=meter.create_counter("gen_ai.bedrock.prompt_caching"),
    )


def _duration_sum(reader: InMemoryMetricReader) -> float:
    for resource_metrics in reader.get_metrics_data().resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == Meters.LLM_OPERATION_DURATION:
                    return sum(data_point.sum for data_point in metric.data.data_points)
    raise AssertionError("operation duration metric not recorded")


def test_operation_duration_measures_the_call_not_the_client_age():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter(__name__)

    metric_params = _metric_params(meter)
    # Simulate a client that was created an hour ago: start_time is the
    # client creation timestamp.
    metric_params.start_time = time.time() - 3600

    def fake_converse(**kwargs: Any) -> dict:
        return CONVERSE_RESPONSE

    wrapped = _instrumented_converse(fake_converse, TracerProvider().get_tracer(__name__), metric_params, None)
    wrapped(modelId=MODEL_ID, messages=[{"role": "user", "content": [{"text": "ping"}]}])

    # The call takes microseconds; a client-age based value is ~3600 seconds.
    assert _duration_sum(reader) < 10
