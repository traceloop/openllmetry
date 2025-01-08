import json
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from opentelemetry.semconv.ai import SpanAttributes
from opentelemetry.trace import get_tracer_provider, Span
from opentelemetry.sdk.trace import ReadableSpan

from opentelemetry.instrumentation.langchain import LangchainInstrumentor

@pytest.fixture
def tracer():
    return get_tracer_provider().get_tracer("test_tracer")

def get_span_events(span: ReadableSpan, event_name: str):
    return [event for event in span.events if event.name == event_name]

def get_span_attribute(span: ReadableSpan, attribute_name: str):
    return span.attributes.get(attribute_name)

def get_span_attributes_by_prefix(span: ReadableSpan, prefix: str):
    return {k: v for k, v in span.attributes.items() if k.startswith(prefix)}

class TestLegacyLangchainEvents:
    def test_llm_completion_legacy_attributes_cohere(
        self, test_context, langchain_use_legacy_attributes_fixture, cohere_llm
    ):
        exporter, _, _ = test_context

        prompt = "Write me a poem about OTel."
        cohere_llm.invoke(prompt)

        spans = exporter.spans
        assert len(spans) == 1
        span = spans[0]

        if langchain_use_legacy_attributes_fixture:
            assert (
                get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content")
                == prompt
            )
            assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"
            assert (
                get_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
                is not None
            )
        else:
            assert not get_span_attribute(
                span, f"{SpanAttributes.LLM_PROMPTS}.0.content"
            )
            assert not get_span_attribute(
                span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content"
            )

class TestNewLangchainEvents:
    def test_llm_completion_new_events_cohere(
        self, test_context, langchain_use_legacy_attributes_fixture, cohere_llm
    ):
        exporter, _, _ = test_context

        prompt = "Write me a poem about OTel."
        output = cohere_llm.invoke(prompt)

        spans = exporter.spans
        assert len(spans) == 1
        span = spans[0]

        if not langchain_use_legacy_attributes_fixture:
            prompt_events = get_span_events(span, "llm.prompt")
            assert len(prompt_events) == 1
            assert prompt_events[0].attributes.get("messaging.role") == "user"
            assert prompt_events[0].attributes.get("messaging.content") == prompt
            assert prompt_events[0].attributes.get("messaging.index") == 0

            completion_events = get_span_events(span, "llm.completion")
            assert len(completion_events) == 1
            assert completion_events[0].attributes.get("messaging.content") == output
            assert completion_events[0].attributes.get("messaging.index") == 0
        else:
            assert not get_span_events(span, "llm.prompt")
            assert not get_span_events(span, "llm.completion")