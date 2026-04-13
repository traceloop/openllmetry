"""Unit tests for custom_llm_instrumentor semconv migration."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

from opentelemetry.instrumentation.llamaindex.custom_llm_instrumentor import (
    _handle_request,
    _handle_response,
)

PATCH_SHOULD_SEND = "opentelemetry.instrumentation.llamaindex.custom_llm_instrumentor.should_send_prompts"


def _span():
    return MagicMock()


def _instance(class_name="Ollama", model_name="llama3", context_window=4096, num_output=512):
    inst = type(class_name, (), {})()
    inst.metadata = SimpleNamespace(model_name=model_name, context_window=context_window, num_output=num_output)
    return inst


def _attr(span, name):
    for call in span.set_attribute.call_args_list:
        if call.args[0] == name:
            return call.args[1]
    return None


def _has_attr(span, name):
    return any(c.args[0] == name for c in span.set_attribute.call_args_list)


# ===========================================================================
# _handle_request
# ===========================================================================

class TestCustomLLMHandleRequest:
    def test_sets_operation_name_chat(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        _handle_request(span, LLMRequestTypeValues.CHAT, (), {}, inst)
        assert _attr(span, GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"

    def test_sets_operation_name_completion(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        _handle_request(span, LLMRequestTypeValues.COMPLETION, (), {}, inst)
        assert _attr(span, GenAIAttributes.GEN_AI_OPERATION_NAME) == "text_completion"

    def test_sets_provider_name(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance("Cohere")
        _handle_request(span, LLMRequestTypeValues.CHAT, (), {}, inst)
        assert _attr(span, GenAIAttributes.GEN_AI_PROVIDER_NAME) == "cohere"

    def test_no_legacy_gen_ai_system(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        _handle_request(span, LLMRequestTypeValues.CHAT, (), {}, inst)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_SYSTEM)

    def test_no_legacy_llm_request_type(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        _handle_request(span, LLMRequestTypeValues.CHAT, (), {}, inst)
        assert not _has_attr(span, SpanAttributes.LLM_REQUEST_TYPE)

    def test_sets_input_messages_json_for_completion(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        with patch(PATCH_SHOULD_SEND, return_value=True):
            _handle_request(span, LLMRequestTypeValues.COMPLETION, ("hello world",), {}, inst)
        raw = _attr(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert raw is not None
        msgs = json.loads(raw)
        assert msgs[0]["role"] == "user"
        assert msgs[0]["parts"][0]["content"] == "hello world"

    def test_gated_by_should_send_prompts(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        with patch(PATCH_SHOULD_SEND, return_value=False):
            _handle_request(span, LLMRequestTypeValues.COMPLETION, ("hello",), {}, inst)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES)

    def test_sets_model(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance(model_name="llama3-70b")
        _handle_request(span, LLMRequestTypeValues.CHAT, (), {}, inst)
        assert _attr(span, GenAIAttributes.GEN_AI_REQUEST_MODEL) == "llama3-70b"

    def test_sets_max_tokens_from_num_output(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance(num_output=1024)
        _handle_request(span, LLMRequestTypeValues.CHAT, (), {}, inst)
        assert _attr(span, GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS) == 1024

    def test_no_top_p_from_num_output(self):
        """num_output should NOT be set as top_p — it maps to max_tokens."""
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance(num_output=512)
        _handle_request(span, LLMRequestTypeValues.CHAT, (), {}, inst)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_REQUEST_TOP_P)

    def test_sets_input_messages_json_for_chat(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        msgs = [SimpleNamespace(role=SimpleNamespace(value="user"), content="Hello", additional_kwargs={})]
        with patch(PATCH_SHOULD_SEND, return_value=True):
            _handle_request(span, LLMRequestTypeValues.CHAT, (msgs,), {}, inst)
        raw = _attr(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert raw is not None
        parsed = json.loads(raw)
        assert parsed[0]["role"] == "user"
        assert parsed[0]["parts"][0]["content"] == "Hello"

    def test_chat_input_gated_by_should_send_prompts(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        msgs = [SimpleNamespace(role=SimpleNamespace(value="user"), content="Hello", additional_kwargs={})]
        with patch(PATCH_SHOULD_SEND, return_value=False):
            _handle_request(span, LLMRequestTypeValues.CHAT, (msgs,), {}, inst)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES)


# ===========================================================================
# _handle_response
# ===========================================================================

class TestCustomLLMHandleResponse:
    def test_sets_output_messages_json_for_completion(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        resp = SimpleNamespace(text="The answer is 42.", raw=None)
        with patch(PATCH_SHOULD_SEND, return_value=True):
            _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        raw = _attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        assert raw is not None
        msgs = json.loads(raw)
        assert msgs[0]["parts"][0]["content"] == "The answer is 42."

    def test_sets_output_messages_json_for_chat(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        msg = SimpleNamespace(role=SimpleNamespace(value="assistant"), content="Reply", additional_kwargs={})
        resp = SimpleNamespace(text="Reply", message=msg, raw=None)
        with patch(PATCH_SHOULD_SEND, return_value=True):
            _handle_response(span, LLMRequestTypeValues.CHAT, inst, resp)
        raw = _attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        assert raw is not None
        msgs = json.loads(raw)
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["parts"][0]["content"] == "Reply"

    def test_sets_response_model(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance(model_name="llama3")
        resp = SimpleNamespace(text="ok", raw=None)
        _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL) == "llama3"

    def test_response_model_prefers_raw_over_instance(self):
        """gen_ai.response.model should use the resolved model from raw response, not the request alias."""
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance(model_name="gpt-4o")
        raw = SimpleNamespace(model="gpt-4o-2024-05-13", id=None)
        resp = SimpleNamespace(text="ok", raw=raw)
        _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL) == "gpt-4o-2024-05-13"

    def test_response_model_falls_back_to_instance_when_no_raw(self):
        """When raw is None, fall back to instance.metadata.model_name."""
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance(model_name="llama3")
        resp = SimpleNamespace(text="ok", raw=None)
        _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL) == "llama3"

    def test_response_model_falls_back_to_instance_when_raw_has_no_model(self):
        """When raw exists but has no model field, fall back to instance.metadata.model_name."""
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance(model_name="ollama-llama3")
        resp = SimpleNamespace(text="ok", raw=SimpleNamespace(id="resp-1"))
        _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL) == "ollama-llama3"

    def test_output_gated_by_should_send_prompts(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        resp = SimpleNamespace(text="ok", raw=None)
        with patch(PATCH_SHOULD_SEND, return_value=False):
            _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)


    def test_sets_response_id_from_raw(self):
        """gen_ai.response.id should be extracted from raw response."""
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        resp = SimpleNamespace(text="ok", raw=SimpleNamespace(id="chatcmpl-abc123", model="llama3"))
        with patch(PATCH_SHOULD_SEND, return_value=True):
            _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_ID) == "chatcmpl-abc123"

    def test_sets_response_id_from_raw_dict(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        resp = SimpleNamespace(text="ok", raw={"id": "resp-456", "model": "llama3"})
        with patch(PATCH_SHOULD_SEND, return_value=True):
            _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_ID) == "resp-456"

    def test_no_response_id_when_missing(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        resp = SimpleNamespace(text="ok", raw={})
        with patch(PATCH_SHOULD_SEND, return_value=True):
            _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_RESPONSE_ID)

    def test_sets_token_usage_from_raw(self):
        """gen_ai.usage.* tokens should be extracted from raw response."""
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        raw = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            id=None,
        )
        resp = SimpleNamespace(text="ok", raw=raw)
        with patch(PATCH_SHOULD_SEND, return_value=True):
            _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        assert _attr(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 10
        assert _attr(span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 20
        assert _attr(span, SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS) == 30

    def test_no_token_usage_when_missing(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        resp = SimpleNamespace(text="ok", raw={})
        with patch(PATCH_SHOULD_SEND, return_value=True):
            _handle_response(span, LLMRequestTypeValues.COMPLETION, inst, resp)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)

    def test_sets_tool_definitions_from_kwargs(self):
        """gen_ai.tool.definitions should be captured from kwargs when tools are passed."""
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
        with patch(PATCH_SHOULD_SEND, return_value=True):
            _handle_request(span, LLMRequestTypeValues.CHAT, (), {"tools": tools}, inst)
        raw = _attr(span, GenAIAttributes.GEN_AI_TOOL_DEFINITIONS)
        assert raw is not None
        parsed = json.loads(raw)
        assert len(parsed) == 1
        assert parsed[0]["function"]["name"] == "get_weather"

    def test_no_tool_definitions_when_not_present(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        with patch(PATCH_SHOULD_SEND, return_value=True):
            _handle_request(span, LLMRequestTypeValues.CHAT, (), {}, inst)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_TOOL_DEFINITIONS)

    def test_tool_definitions_gated_by_should_send_prompts(self):
        from opentelemetry.semconv_ai import LLMRequestTypeValues
        span = _span()
        inst = _instance()
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        with patch(PATCH_SHOULD_SEND, return_value=False):
            _handle_request(span, LLMRequestTypeValues.CHAT, (), {"tools": tools}, inst)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_TOOL_DEFINITIONS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
