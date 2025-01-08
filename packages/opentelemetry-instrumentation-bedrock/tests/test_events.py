import json
from unittest.mock import patch

import pytest

from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import get_tracer_provider, Span
from opentelemetry.sdk.trace import ReadableSpan

from opentelemetry.instrumentation.bedrock import BedrockInstrumentor

@pytest.fixture
def tracer():
    return get_tracer_provider().get_tracer("test_tracer")

def get_span_events(span: ReadableSpan, event_name: str):
    return [event for event in span.events if event.name == event_name]

def get_span_attribute(span: ReadableSpan, attribute_name: str):
    return span.attributes.get(attribute_name)

def get_span_attributes_by_prefix(span: ReadableSpan, prefix: str):
    return {k: v for k, v in span.attributes.items() if k.startswith(prefix)}

class TestLegacyBedrockEvents:
    def test_completion_legacy_attributes(self, brt, test_context, use_legacy_attributes_fixture):
        exporter, _, _ = test_context

        body = {
            "inputText": "Write me a poem about OTel.",
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.7,
                "topP": 0.9,
            }
        }
        try:
            response = brt.invoke_model(
                modelId="amazon.titan-text-express-v1",
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )
            response_body = json.loads(response.get("body").read())
        except Exception as e:
            print(f"Error invoking model: {e}")  # Handle the exception properly
            response_body = {}

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        if use_legacy_attributes_fixture:
            assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content") == "Write me a poem about OTel."
            assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"
            assert get_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response_body.get("results", [{}])[0].get("outputText")
        else:
            assert not get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content")
            assert not get_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content")

class TestNewBedrockEvents:
    def test_completion_new_events(self, brt, test_context, use_legacy_attributes_fixture):
        exporter, _, _ = test_context

        body = {
            "inputText": "Write me a poem about OTel.",
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.7,
                "topP": 0.9,
            }
        }
        try:
            response = brt.invoke_model(
                modelId="amazon.titan-text-express-v1",
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )
            response_body = json.loads(response.get("body").read())

        except Exception as e:
            print(f"Error invoking model: {e}")
            response_body = {}

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        if not use_legacy_attributes_fixture:
            prompt_events = get_span_events(span, "prompt")
            assert len(prompt_events) == 1
            assert prompt_events[0].attributes.get("messaging.role") == "user"
            assert prompt_events[0].attributes.get("messaging.content") == "Write me a poem about OTel."
            assert prompt_events[0].attributes.get("messaging.index") == 0

            completion_events = get_span_events(span, "completion")
            assert len(completion_events) == 1
            assert completion_events[0].attributes.get("messaging.content") == response_body.get("results", [{}])[0].get("outputText")
            assert completion_events[0].attributes.get("messaging.index") == 0
        else:
            assert not get_span_events(span, "prompt")
            assert not get_span_events(span, "completion")

    def test_chat_legacy_attributes(self, brt, test_context, use_legacy_attributes_fixture):
        exporter, _, _ = test_context
        # Titan Text Express does not have a specific "chat" mode like Claude.
        # We can simulate a chat interaction with a single turn.
        body = {
            "inputText": "User: What is the meaning of life?\nAssistant:",
            "textGenerationConfig": {
                "maxTokenCount": 256,
                "temperature": 0.7,
                "topP": 0.9,
            }
        }
        try:
            response = brt.invoke_model(
                modelId="amazon.titan-text-express-v1",
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )
            response_body = json.loads(response.get("body").read())

        except Exception as e:
            print(f"Error invoking model: {e}")
            response_body = {}

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        if use_legacy_attributes_fixture:
            assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content") == "User: What is the meaning of life?\nAssistant:"
            assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"
            assert get_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response_body.get("results", [{}])[0].get("outputText")
            # We can't assert the assistant role in this case because Titan Text Express doesn't use that concept.
        else:
             assert not get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content")
             assert not get_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content")

    def test_chat_new_events(self, brt, test_context, use_legacy_attributes_fixture):
        exporter, _, _ = test_context

        body = {
            "inputText": "User: What is the meaning of life?\nAssistant:",
            "textGenerationConfig": {
                "maxTokenCount": 256,
                "temperature": 0.7,
                "topP": 0.9,
            }
        }
        try:
            response = brt.invoke_model(
                modelId="amazon.titan-text-express-v1",
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )
            response_body = json.loads(response.get("body").read())
        except Exception as e:
            print(f"Error invoking model: {e}")
            response_body = {}
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        if not use_legacy_attributes_fixture:
            prompt_events = get_span_events(span, "prompt")
            assert len(prompt_events) == 1
            assert prompt_events[0].attributes.get("messaging.role") == "user"
            assert prompt_events[0].attributes.get("messaging.content") == "User: What is the meaning of life?\nAssistant:"
            assert prompt_events[0].attributes.get("messaging.index") == 0

            completion_events = get_span_events(span, "completion")
            assert len(completion_events) == 1
            assert completion_events[0].attributes.get("messaging.content") == response_body.get("results", [{}])[0].get("outputText")
            assert completion_events[0].attributes.get("messaging.index") == 0
        else:
            assert not get_span_events(span, "prompt")
            assert not get_span_events(span, "completion")

    def test_streaming_legacy_attributes(self, brt, test_context, use_legacy_attributes_fixture):
        exporter, _, _ = test_context

        body = {
            "inputText": "Tell me a joke about OTel",
            "textGenerationConfig": {
                "maxTokenCount": 256,
                "temperature": 0.7,
                "topP": 0.9,
            }
        }
        try:
            response = brt.invoke_model_with_response_stream(
                modelId="amazon.titan-text-express-v1", body=json.dumps(body)
            )
            for event in response.get('body'):
                pass

        except Exception as e:
            print(f"Error invoking model: {e}")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        if use_legacy_attributes_fixture:
            assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content") == "Tell me a joke about OTel"
            assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"
        else:
            assert not get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content")
            assert not get_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content")
        # Asserting completion content in streaming might require inspecting the response events.

    def test_streaming_new_events(self, brt, test_context, use_legacy_attributes_fixture):
        exporter, _, _ = test_context

        body = {
            "inputText": "Tell me a joke about OTel",
            "textGenerationConfig": {
                "maxTokenCount": 256,
                "temperature": 0.7,
                "topP": 0.9,
            }
        }
        try:
            response = brt.invoke_model_with_response_stream(
                modelId="amazon.titan-text-express-v1", body=json.dumps(body)
            )
            for event in response.get('body'):
                pass
        except Exception as e:
            print(f"Error invoking model: {e}")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        if not use_legacy_attributes_fixture:
            prompt_events = get_span_events(span, "prompt")
            assert len(prompt_events) == 1
            assert prompt_events[0].attributes.get("messaging.role") == "user"
            # For streaming, the prompt content might be sent in the initial event.
            # We perform a basic check to ensure some part of the prompt is captured.
            assert prompt_events[0].attributes.get("messaging.content") is not None
            assert "Tell me a joke about OTel" in prompt_events[0].attributes.get("messaging.content")
            assert prompt_events[0].attributes.get("messaging.index") == 0

            completion_events = get_span_events(span, "completion")
            assert len(completion_events) >= 1  # Can be multiple completion events in streaming
        else:
            assert not get_span_events(span, "prompt")
            assert not get_span_events(span, "completion")

    def test_streaming_chat_legacy_attributes(self, brt, test_context, use_legacy_attributes_fixture):
        exporter, _, _ = test_context

        body = {
            "inputText": "User: Explain the benefits of using OpenTelemetry.\nAssistant:",
            "textGenerationConfig": {
                "maxTokenCount": 256,
                "temperature": 0.7,
                "topP": 0.9,
            }
        }
        try:
            response = brt.invoke_model_with_response_stream(
                modelId="amazon.titan-text-express-v1", body=json.dumps(body)
            )
            for event in response.get('body'):
                pass
        except Exception as e:
            print(f"Error invoking model: {e}")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        if use_legacy_attributes_fixture:
            assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content") == "User: Explain the benefits of using OpenTelemetry.\nAssistant:"
            assert get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.role") == "user"
            # Asserting completion content in streaming might require inspecting the response events.
        else:
            assert not get_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.content")
            assert not get_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content")

    def test_streaming_chat_new_events(self, brt, test_context, use_legacy_attributes_fixture):
        exporter, _, _ = test_context

        body = {
            "inputText": "User: Explain the benefits of using OpenTelemetry.\nAssistant:",
            "textGenerationConfig": {
                "maxTokenCount": 256,
                "temperature": 0.7,
                "topP": 0.9,
            }
        }
        try:
            response = brt.invoke_model_with_response_stream(
                modelId="amazon.titan-text-express-v1", body=json.dumps(body)
            )
            for event in response.get('body'):
                pass
        except Exception as e:
            print(f"Error invoking model: {e}")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        if not use_legacy_attributes_fixture:
            prompt_events = get_span_events(span, "prompt")
            assert len(prompt_events) == 1
            assert prompt_events[0].attributes.get("messaging.role") == "user"
            # For streaming, the prompt content might be sent in the initial event.
            # We perform a basic check to ensure some part of the prompt is captured.
            assert prompt_events[0].attributes.get("messaging.content") is not None
            assert prompt_events[0].attributes.get("messaging.index") == 0

            completion_events = get_span_events(span, "completion")
            assert len(completion_events) >= 1
        else:
            assert not get_span_events(span, "prompt")
            assert not get_span_events(span, "completion")