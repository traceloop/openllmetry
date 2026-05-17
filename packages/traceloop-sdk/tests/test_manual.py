import pytest
from openai import OpenAI
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from traceloop.sdk.tracing.manual import LLMMessage, LLMUsage, track_llm_call


@pytest.fixture
def openai_client():
    return OpenAI()


def test_manual_report(exporter, openai_client):
    with track_llm_call(vendor="openai", type="chat") as span:
        span.report_request(
            model="gpt-3.5-turbo",
            messages=[
                LLMMessage(role="user", content="Tell me a joke about opentelemetry")
            ],
        )

        res = [
            "Why did the opentelemetry developer break up with their partner? Because they were tired"
            + " of constantly tracing their every move!",
        ]

        span.report_response("gpt-3.5-turbo-0125", res)
        span.report_usage(
            LLMUsage(
                prompt_tokens=15,
                completion_tokens=24,
                total_tokens=39,
                cache_creation_input_tokens=15,
                cache_read_input_tokens=18,
            )
        )

    spans = exporter.get_finished_spans()
    open_ai_span = spans[0]
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "user"
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == "gpt-3.5-turbo-0125"
    )
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == "Why did the opentelemetry developer break up with their partner? Because they were tired"
        + " of constantly tracing their every move!"
    )
    assert open_ai_span.end_time > open_ai_span.start_time
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 15
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 24
    assert open_ai_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 39
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_USAGE_CACHE_CREATION_INPUT_TOKENS]
        == 15
    )
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS] == 18
    )
