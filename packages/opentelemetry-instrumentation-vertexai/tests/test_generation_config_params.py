import pytest
from opentelemetry.instrumentation.vertexai.span_utils import set_model_input_attributes
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from vertexai.generative_models import GenerationConfig

MODEL = "gemini-2.5-flash"

ALL_PARAMS = {
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 3,
    "max_output_tokens": 50,
    "presence_penalty": 0.1,
    "frequency_penalty": 0.2,
}

SAMPLING_ATTRIBUTES = [
    GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE,
    GenAIAttributes.GEN_AI_REQUEST_TOP_P,
    GenAIAttributes.GEN_AI_REQUEST_TOP_K,
    GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS,
    SpanAttributes.LLM_PRESENCE_PENALTY,
    SpanAttributes.LLM_FREQUENCY_PENALTY,
]


def _request_attributes(tracer_provider, span_exporter, kwargs):
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("generate_content") as span:
        set_model_input_attributes(span, kwargs, MODEL)
    (finished,) = span_exporter.get_finished_spans()
    return finished.attributes


def _assert_all_params(attrs):
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == pytest.approx(0.2)
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == pytest.approx(0.8)
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 3
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 50
    assert attrs[SpanAttributes.LLM_PRESENCE_PENALTY] == pytest.approx(0.1)
    assert attrs[SpanAttributes.LLM_FREQUENCY_PENALTY] == pytest.approx(0.2)


def test_generation_config_object(tracer_provider, span_exporter):
    config = GenerationConfig(**ALL_PARAMS)

    attrs = _request_attributes(tracer_provider, span_exporter, {"generation_config": config})

    _assert_all_params(attrs)


def test_generation_config_dict(tracer_provider, span_exporter):
    attrs = _request_attributes(tracer_provider, span_exporter, {"generation_config": dict(ALL_PARAMS)})

    _assert_all_params(attrs)


def test_legacy_kwargs_still_recorded(tracer_provider, span_exporter):
    attrs = _request_attributes(tracer_provider, span_exporter, dict(ALL_PARAMS))

    _assert_all_params(attrs)


def test_zero_temperature_recorded_without_unset_params(tracer_provider, span_exporter):
    config = GenerationConfig(temperature=0.0)

    attrs = _request_attributes(tracer_provider, span_exporter, {"generation_config": config})

    assert attrs[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.0
    assert GenAIAttributes.GEN_AI_REQUEST_TOP_P not in attrs
    assert GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS not in attrs
    assert GenAIAttributes.GEN_AI_REQUEST_TOP_K not in attrs
    assert SpanAttributes.LLM_PRESENCE_PENALTY not in attrs
    assert SpanAttributes.LLM_FREQUENCY_PENALTY not in attrs


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"generation_config": None}, {"generation_config": GenerationConfig()}],
)
def test_absent_or_empty_generation_config_records_nothing(tracer_provider, span_exporter, kwargs):
    attrs = _request_attributes(tracer_provider, span_exporter, kwargs)

    for name in SAMPLING_ATTRIBUTES:
        assert name not in attrs
