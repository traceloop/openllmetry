from opentelemetry.semconv.ai import SpanAttributes
import pytest
from openai import OpenAI
from traceloop.sdk.tracing.manual import LLMMessage, track_llm_call


@pytest.fixture
def openai_client():
    return OpenAI()


@pytest.mark.vcr
def test_manual_report(exporter, openai_client):
    with track_llm_call(vendor="openai", type="chat") as span:
        span.report_request(
            model="gpt-3.5-turbo",
            messages=[
                LLMMessage(role="user", content="Tell me a joke about opentelemetry")
            ],
        )

        res = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )

        span.report_response(res.model, [text.message.content for text in res.choices])

    spans = exporter.get_finished_spans()
    open_ai_span = spans[0]
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert (
        open_ai_span.attributes[SpanAttributes.LLM_RESPONSE_MODEL]
        == "gpt-3.5-turbo-0125"
    )
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == "Why did the opentelemetry developer break up with their partner? Because they were tired"
        + " of constantly tracing their every move!"
    )
    assert open_ai_span.end_time > open_ai_span.start_time
