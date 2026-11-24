import pytest
from opentelemetry.instrumentation.together.span_utils import (
    set_model_prompt_attributes,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

REQUEST_PARAMETER_ATTRIBUTES = (
    GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS,
    GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE,
    GenAIAttributes.GEN_AI_REQUEST_TOP_P,
    GenAIAttributes.GEN_AI_REQUEST_TOP_K,
    GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY,
    SpanAttributes.GEN_AI_REQUEST_REPETITION_PENALTY,
    GenAIAttributes.GEN_AI_REQUEST_SEED,
    GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES,
)


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(__name__)


def _record(tracer, kwargs):
    span = tracer.start_span("together.chat")
    set_model_prompt_attributes(span, kwargs)
    span.end()
    return span.attributes


def test_all_request_parameters_are_recorded(tracer):
    attributes = _record(
        tracer,
        {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "max_tokens": 64,
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.1,
            "repetition_penalty": 1.1,
            "seed": 7,
            "stop": ["END", "STOP"],
        },
    )

    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 64
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.3
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.8
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY] == 0.2
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY] == 0.1
    assert attributes[SpanAttributes.GEN_AI_REQUEST_REPETITION_PENALTY] == 1.1
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_SEED] == 7
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES] == (
        "END",
        "STOP",
    )


def test_omitted_request_parameters_are_not_recorded(tracer):
    attributes = _record(
        tracer, {"model": "mistralai/Mixtral-8x7B-Instruct-v0.1"}
    )

    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == (
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    for name in REQUEST_PARAMETER_ATTRIBUTES:
        assert name not in attributes


def test_string_stop_is_recorded_as_single_sequence(tracer):
    attributes = _record(tracer, {"stop": "END"})

    assert attributes[GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES] == ("END",)


def test_empty_string_stop_is_not_recorded(tracer):
    attributes = _record(tracer, {"stop": ""})

    assert GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES not in attributes


def test_empty_stop_is_not_recorded(tracer):
    attributes = _record(tracer, {"stop": []})

    assert GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES not in attributes


def test_zero_valued_parameters_are_recorded(tracer):
    attributes = _record(tracer, {"temperature": 0.0, "seed": 0, "top_k": 0})

    assert attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.0
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_SEED] == 0
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 0
