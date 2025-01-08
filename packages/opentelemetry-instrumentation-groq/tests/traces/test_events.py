

import json
from unittest.mock import patch

import pytest

from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import get_tracer_provider, Span
from opentelemetry.sdk.trace import ReadableSpan

from opentelemetry.instrumentation.groq import GroqInstrumentor
from opentelemetry.instrumentation.groq.config import Config

def get_span_events(span: ReadableSpan, event_name: str):
    return [event for event in span.events if event.name == event_name]

def get_span_attribute(span: ReadableSpan, attribute_name: str):
    return span.attributes.get(attribute_name)

def get_span_attributes_by_prefix(span: ReadableSpan, prefix: str):
    return {k: v for k, v in span.attributes.items() if k.startswith(prefix)}

class TestGroqEvents:

    def test_completion_legacy_attributes(self, groq_client, exporter, instrument):
        Config.use_legacy_attributes = True
        prompt = "Write a short poem about OTel"
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
        )
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content") == prompt
        assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"
        assert get_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response.choices[0].message.content

    def test_completion_new_events(self, groq_client, exporter, instrument):
        Config.use_legacy_attributes = False
        prompt = "Write a haiku about tracing"
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
        )
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        prompt_events = get_span_events(span, "prompt")
        print(f"Prompt Events: {[event.attributes for event in prompt_events]}")
        assert len(prompt_events) == 1
        assert prompt_events[0].attributes.get("messaging.role") == "user"
        assert prompt_events[0].attributes.get("messaging.content") == prompt
        assert prompt_events[0].attributes.get("messaging.index") == 0

        completion_events = get_span_events(span, "completion")
        print(f"Completion Events: {[event.attributes for event in completion_events]}")  # Add this

        assert len(completion_events) == 1
        assert completion_events[0].attributes.get("messaging.content") == response.choices[0].message.content
        assert completion_events[0].attributes.get("messaging.index") == 0
    
    @pytest.mark.asyncio
    async def test_async_completion_legacy_attributes(self, async_groq_client, exporter, instrument):
        Config.use_legacy_attributes = True
        prompt = "Explain the benefits of asynchronous programming"
        response = await async_groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
        )
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content") == prompt
        assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"
        assert get_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response.choices[0].message.content

    @pytest.mark.asyncio
    async def test_async_completion_new_events(self, async_groq_client, exporter, instrument):
        Config.use_legacy_attributes = False
        prompt = "Describe the concept of a microservice"
        response = await async_groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
        )
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        prompt_events = get_span_events(span, "prompt")
        print(f"Prompt Events: {[event.attributes for event in prompt_events]}")
        assert len(prompt_events) == 1
        assert prompt_events[0].attributes.get("messaging.role") == "user"
        assert prompt_events[0].attributes.get("messaging.content") == prompt
        assert prompt_events[0].attributes.get("messaging.index") == 0

        completion_events = get_span_events(span, "completion")
        print(f"Completion Events: {[event.attributes for event in completion_events]}")  # Add this

        assert len(completion_events) == 1
        assert completion_events[0].attributes.get("messaging.content") == response.choices[0].message.content
        assert completion_events[0].attributes.get("messaging.index") == 0
 
    def test_chat_legacy_attributes(self, groq_client, exporter, instrument):
        Config.use_legacy_attributes = True
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "What is the capital of Germany?"},
        ]
        response = groq_client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",
        )
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"
        assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content") == "What is the capital of France?"
        assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.1.role") == "assistant"
        assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.1.content") == "Paris."
        assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.2.role") == "user"
        assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.2.content") == "What is the capital of Germany?"
        assert get_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response.choices[0].message.content
        assert get_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"

    
    def test_chat_new_events(self, groq_client, exporter, instrument):
        Config.use_legacy_attributes = False
        messages = [
            {"role": "user", "content": "Explain the theory of relativity."},
            {"role": "assistant", "content": "It's a complex topic..."},
            {"role": "user", "content": "Simplify it for a beginner."},
        ]
        response = groq_client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",
        )
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        prompt_events = get_span_events(span, "prompt")
        print(f"Prompt Events: {[event.attributes for event in prompt_events]}")
        assert len(prompt_events) == 3
        assert prompt_events[0].attributes.get("messaging.role") == "user"
        assert prompt_events[0].attributes.get("messaging.content") == "Explain the theory of relativity."
        assert prompt_events[0].attributes.get("messaging.index") == 0
        assert prompt_events[1].attributes.get("messaging.role") == "assistant"
        assert prompt_events[1].attributes.get("messaging.content") == "It's a complex topic..."
        assert prompt_events[1].attributes.get("messaging.index") == 1
        assert prompt_events[2].attributes.get("messaging.role") == "user"
        assert prompt_events[2].attributes.get("messaging.content") == "Simplify it for a beginner."
        assert prompt_events[2].attributes.get("messaging.index") == 2

        completion_events = get_span_events(span, "completion")
        print(f"Completion Events: {[event.attributes for event in completion_events]}")  # Add this

        assert len(completion_events) == 1
        assert completion_events[0].attributes.get("messaging.content") == response.choices[0].message.content
        assert completion_events[0].attributes.get("messaging.index") == 0
    
    @pytest.mark.asyncio
    async def test_async_chat_legacy_attributes(self, async_groq_client, exporter, instrument):
        Config.use_legacy_attributes = True
        messages = [
            {"role": "user", "content": "What are the main principles of OOP?"},
            {"role": "assistant", "content": "Encapsulation, inheritance, and polymorphism."},
            {"role": "user", "content": "Explain encapsulation."},
        ]
        response = await async_groq_client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",
        )
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"
        assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content") == "What are the main principles of OOP?"
        assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.1.role") == "assistant"
        assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.1.content") == "Encapsulation, inheritance, and polymorphism."
        assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.2.role") == "user"
        assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.2.content") == "Explain encapsulation."
        assert get_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response.choices[0].message.content
        assert get_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.role") == "assistant"
    
    @pytest.mark.asyncio
    async def test_async_chat_new_events(self, async_groq_client, exporter, instrument):
        Config.use_legacy_attributes = False
        messages = [
            {"role": "user", "content": "Define cloud computing."},
            {"role": "assistant", "content": "It's the delivery of computing services..."},
            {"role": "user", "content": "Give some examples of cloud services."},
        ]
        response = await async_groq_client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",
        )
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        prompt_events = get_span_events(span, "prompt")
        print(f"Prompt Events: {[event.attributes for event in prompt_events]}")
        assert len(prompt_events) == 3
        assert prompt_events[0].attributes.get("messaging.role") == "user"
        assert prompt_events[0].attributes.get("messaging.content") == "Define cloud computing."
        assert prompt_events[0].attributes.get("messaging.index") == 0
        assert prompt_events[1].attributes.get("messaging.role") == "assistant"
        assert prompt_events[1].attributes.get("messaging.content") == "It's the delivery of computing services..."
        assert prompt_events[1].attributes.get("messaging.index") == 1
        assert prompt_events[2].attributes.get("messaging.role") == "user"
        assert prompt_events[2].attributes.get("messaging.content") == "Give some examples of cloud services."
        assert prompt_events[2].attributes.get("messaging.index") == 2

        completion_events = get_span_events(span, "completion")
        print(f"Completion Events: {[event.attributes for event in completion_events]}")  # Add this

        assert len(completion_events) == 1
        assert completion_events[0].attributes.get("messaging.content") == response.choices[0].message.content
        assert completion_events[0].attributes.get("messaging.index") == 0