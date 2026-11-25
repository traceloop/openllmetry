from opentelemetry import trace
from opentelemetry.instrumentation.bedrock import (
    MetricParams,
    _get_vendor_model,
    _handle_converse,
    _handle_converse_stream,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import Meters

MODEL_ID = "amazon.titan-text-express-v1"
# The label the metric carries is whatever `_get_vendor_model` derives from the model id,
# so derive the expectation the same way instead of duplicating the parsing rule here.
_, _, EXPECTED_MODEL = _get_vendor_model(MODEL_ID)
STALE_VENDOR = "anthropic"
STALE_MODEL = "anthropic.claude-3-5-sonnet"

CONVERSE_RESPONSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 5, "outputTokens": 7, "totalTokens": 12},
}

CONVERSE_KWARGS = {
    "modelId": MODEL_ID,
    "messages": [{"role": "user", "content": [{"text": "hi"}]}],
}


def _metric_params():
    """Real metric instruments backed by an in-memory reader."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("test")

    metric_params = MetricParams(
        token_histogram=meter.create_histogram(Meters.LLM_TOKEN_USAGE),
        choice_counter=meter.create_counter(Meters.LLM_GENERATION_CHOICES),
        duration_histogram=meter.create_histogram(Meters.LLM_OPERATION_DURATION),
        exception_counter=meter.create_counter("gen_ai.bedrock.completions.exceptions"),
        guardrail_activation=meter.create_counter("guardrail.activation"),
        guardrail_latency_histogram=meter.create_histogram("guardrail.latency"),
        guardrail_coverage=meter.create_counter("guardrail.coverage"),
        guardrail_sensitive_info=meter.create_counter("guardrail.sensitive_info"),
        guardrail_topic=meter.create_counter("guardrail.topic"),
        guardrail_content=meter.create_counter("guardrail.content"),
        guardrail_words=meter.create_counter("guardrail.words"),
        prompt_caching=meter.create_counter("prompt.caching"),
    )

    # What a previous invoke_model call would have left on the shared params.
    metric_params.vendor = STALE_VENDOR
    metric_params.model = STALE_MODEL
    metric_params.is_stream = False

    return metric_params, reader


def _recorded_models(reader):
    models = []
    for resource_metrics in reader.get_metrics_data().resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                for data_point in metric.data.data_points:
                    models.append(data_point.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL))
    return models


def _span():
    return trace.get_tracer(__name__).start_span("test")


def test_converse_metrics_are_labelled_with_their_own_model():
    """Converse metric points must name the converse model, not a previous call's.

    `metric_params` is shared by every call on the instrumentor and the metric labels are
    read from it, so it has to be updated per call. Without that, a fresh instrumentor
    labelled points with "" and an invoke_model call left its model behind on subsequent
    converse points.
    """
    metric_params, reader = _metric_params()

    _handle_converse(_span(), CONVERSE_KWARGS, CONVERSE_RESPONSE, metric_params, None)

    models = _recorded_models(reader)
    assert models, "expected the converse call to record metric points"
    assert set(models) == {EXPECTED_MODEL}
    assert STALE_MODEL not in models


def test_converse_stream_metrics_are_labelled_with_their_own_model():
    metric_params, reader = _metric_params()

    stream_response = {key: value for key, value in CONVERSE_RESPONSE.items() if key != "usage"}

    def _parse_event(*args, **kwargs):
        return {
            "metadata": {
                "usage": {"inputTokens": 5, "outputTokens": 7},
            }
        }

    stream_response["stream"] = type("Stream", (), {"_parse_event": _parse_event})()

    _handle_converse_stream(_span(), CONVERSE_KWARGS, stream_response, metric_params, None)
    for _ in stream_response["stream"]._parse_event():
        pass

    models = _recorded_models(reader)
    assert models, "expected the streamed converse call to record metric points"
    assert set(models) == {EXPECTED_MODEL}
    assert STALE_MODEL not in models
