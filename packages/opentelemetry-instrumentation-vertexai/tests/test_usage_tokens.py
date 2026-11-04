import pytest
from opentelemetry.instrumentation.vertexai.span_utils import (
    set_model_response_attributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from vertexai.generative_models import GenerationResponse


def _usage(**counts):
    response = GenerationResponse.from_dict(
        {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}],
            "usage_metadata": {"prompt_token_count": 10, **counts},
        }
    )
    return response.usage_metadata


def _span_attributes(tracer_provider, span_exporter, usage):
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("generate_content") as span:
        set_model_response_attributes(span, "gemini-2.5-flash", usage)
    (finished,) = span_exporter.get_finished_spans()
    return finished.attributes


def test_thinking_tokens_count_as_output(tracer_provider, span_exporter):
    usage = _usage(candidates_token_count=2, thoughts_token_count=300, total_token_count=312)

    attrs = _span_attributes(tracer_provider, span_exporter, usage)

    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 302
    assert attrs[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 300
    assert (
        attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] + attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == attrs[SpanAttributes.LLM_USAGE_TOTAL_TOKENS]
    )


@pytest.mark.parametrize(
    "counts, expected_output",
    [
        ({"candidates_token_count": 2}, 2),
        ({"thoughts_token_count": 300}, 300),
    ],
)
def test_missing_token_counts(tracer_provider, span_exporter, counts, expected_output):
    usage = _usage(**counts)

    attrs = _span_attributes(tracer_provider, span_exporter, usage)

    assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == expected_output
    if "thoughts_token_count" in counts:
        assert attrs[SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS] == 300
    else:
        assert SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS not in attrs
