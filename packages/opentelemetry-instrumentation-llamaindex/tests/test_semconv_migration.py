"""Span-level semconv migration tests — verifies span_utils functions set correct attributes."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.base.llms.types import MessageRole

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

from opentelemetry.instrumentation.llamaindex.span_utils import (
    set_embedding,
    set_llm_chat_request,
    set_llm_chat_request_model_attributes,
    set_llm_chat_response,
    set_llm_chat_response_model_attributes,
    set_llm_predict_response,
    set_rerank,
    set_rerank_model_attributes,
)

PATCH_SHOULD_SEND = "opentelemetry.instrumentation.llamaindex.span_utils.should_send_prompts"


def _recording_span():
    span = MagicMock()
    span.is_recording.return_value = True
    return span


def _chat_message(role, content, **additional_kwargs):
    m = MagicMock()
    m.role = MessageRole(role)
    m.content = content
    m.additional_kwargs = additional_kwargs
    return m


def _attr(span, name):
    """Get the value set_attribute was called with for a given attribute name."""
    for call in span.set_attribute.call_args_list:
        if call.args[0] == name:
            return call.args[1]
    return None


def _has_attr(span, name):
    return any(c.args[0] == name for c in span.set_attribute.call_args_list)


# ===========================================================================
# set_llm_chat_request — input messages as JSON
# ===========================================================================

class TestSetLlmChatRequest:
    def test_sets_gen_ai_input_messages_json(self):
        span = _recording_span()
        event = MagicMock()
        event.messages = [_chat_message("user", "Hello")]
        with patch(PATCH_SHOULD_SEND, return_value=True):
            set_llm_chat_request(event, span)
        raw = _attr(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert raw is not None
        msgs = json.loads(raw)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["parts"][0]["content"] == "Hello"

    def test_gated_by_should_send_prompts(self):
        span = _recording_span()
        event = MagicMock()
        event.messages = [_chat_message("user", "Hello")]
        with patch(PATCH_SHOULD_SEND, return_value=False):
            set_llm_chat_request(event, span)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES)

    def test_skips_when_not_recording(self):
        span = MagicMock()
        span.is_recording.return_value = False
        event = MagicMock()
        event.messages = [_chat_message("user", "Hello")]
        with patch(PATCH_SHOULD_SEND, return_value=True):
            set_llm_chat_request(event, span)
        span.set_attribute.assert_not_called()

    def test_multiple_messages_preserved(self):
        span = _recording_span()
        event = MagicMock()
        event.messages = [
            _chat_message("system", "Be helpful"),
            _chat_message("user", "Hi"),
        ]
        with patch(PATCH_SHOULD_SEND, return_value=True):
            set_llm_chat_request(event, span)
        msgs = json.loads(_attr(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES))
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_no_legacy_indexed_attributes(self):
        span = _recording_span()
        event = MagicMock()
        event.messages = [_chat_message("user", "Hello")]
        with patch(PATCH_SHOULD_SEND, return_value=True):
            set_llm_chat_request(event, span)
        for call in span.set_attribute.call_args_list:
            assert not call.args[0].startswith(f"{GenAIAttributes.GEN_AI_PROMPT}.")


# ===========================================================================
# set_llm_chat_request_model_attributes — operation.name + provider.name
# ===========================================================================

class TestSetLlmChatRequestModelAttributes:
    def test_sets_operation_name_chat(self):
        span = _recording_span()
        event = MagicMock()
        event.model_dict = {"model": "gpt-4", "temperature": 0.7}
        set_llm_chat_request_model_attributes(event, span)
        assert _attr(span, GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"

    def test_sets_model(self):
        span = _recording_span()
        event = MagicMock()
        event.model_dict = {"model": "gpt-4"}
        set_llm_chat_request_model_attributes(event, span)
        assert _attr(span, GenAIAttributes.GEN_AI_REQUEST_MODEL) == "gpt-4"

    def test_sets_temperature(self):
        span = _recording_span()
        event = MagicMock()
        event.model_dict = {"model": "gpt-4", "temperature": 0.5}
        set_llm_chat_request_model_attributes(event, span)
        assert _attr(span, GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE) == 0.5

    def test_structured_llm_nested_model(self):
        span = _recording_span()
        event = MagicMock()
        event.model_dict = {"llm": {"model": "gpt-4", "temperature": 0.3}}
        set_llm_chat_request_model_attributes(event, span)
        assert _attr(span, GenAIAttributes.GEN_AI_REQUEST_MODEL) == "gpt-4"

    def test_sets_provider_name_from_class_name(self):
        span = _recording_span()
        event = MagicMock()
        event.model_dict = {"model": "gpt-4", "class_name": "OpenAI"}
        set_llm_chat_request_model_attributes(event, span)
        assert _attr(span, GenAIAttributes.GEN_AI_PROVIDER_NAME) == "openai"

    def test_no_legacy_llm_request_type(self):
        span = _recording_span()
        event = MagicMock()
        event.model_dict = {"model": "gpt-4"}
        set_llm_chat_request_model_attributes(event, span)
        assert not _has_attr(span, SpanAttributes.LLM_REQUEST_TYPE)


# ===========================================================================
# set_llm_chat_response — output messages as JSON
# ===========================================================================

class TestSetLlmChatResponse:
    def test_sets_gen_ai_output_messages_json(self):
        span = _recording_span()
        msg = _chat_message("assistant", "The answer is 42.")
        event = MagicMock()
        event.response = MagicMock(message=msg, raw={"choices": [{"finish_reason": "stop"}]})
        event.messages = [_chat_message("user", "What is 42?")]
        with patch(PATCH_SHOULD_SEND, return_value=True):
            set_llm_chat_response(event, span)
        raw_out = _attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        assert raw_out is not None
        msgs = json.loads(raw_out)
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["parts"][0]["content"] == "The answer is 42."
        assert "finish_reason" in msgs[0]

    def test_does_not_set_input_messages(self):
        """Input messages are set by set_llm_chat_request, not set_llm_chat_response."""
        span = _recording_span()
        msg = _chat_message("assistant", "Reply")
        event = MagicMock()
        event.response = MagicMock(message=msg, raw={})
        event.messages = [_chat_message("user", "Hello")]
        with patch(PATCH_SHOULD_SEND, return_value=True):
            set_llm_chat_response(event, span)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES)

    def test_gated_by_should_send_prompts(self):
        span = _recording_span()
        msg = _chat_message("assistant", "Reply")
        event = MagicMock()
        event.response = MagicMock(message=msg, raw={})
        event.messages = [_chat_message("user", "Hello")]
        with patch(PATCH_SHOULD_SEND, return_value=False):
            set_llm_chat_response(event, span)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)

    def test_none_content_in_output(self):
        span = _recording_span()
        msg = _chat_message("assistant", None)
        event = MagicMock()
        event.response = MagicMock(message=msg, raw={})
        event.messages = []
        with patch(PATCH_SHOULD_SEND, return_value=True):
            set_llm_chat_response(event, span)
        msgs = json.loads(_attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES))
        assert msgs[0]["parts"] == []

    def test_no_legacy_indexed_attributes(self):
        span = _recording_span()
        msg = _chat_message("assistant", "Reply")
        event = MagicMock()
        event.response = MagicMock(message=msg, raw={})
        event.messages = [_chat_message("user", "Hi")]
        with patch(PATCH_SHOULD_SEND, return_value=True):
            set_llm_chat_response(event, span)
        for call in span.set_attribute.call_args_list:
            key = call.args[0]
            assert not key.startswith(f"{GenAIAttributes.GEN_AI_PROMPT}.")
            assert not key.startswith(f"{GenAIAttributes.GEN_AI_COMPLETION}.")

    def test_sets_finish_reasons_span_attr_independently(self):
        """set_llm_chat_response must set gen_ai.response.finish_reasons on its own,
        without relying on set_llm_chat_response_model_attributes."""
        span = _recording_span()
        msg = _chat_message("assistant", "Done.")
        event = MagicMock()
        event.response = MagicMock(
            message=msg,
            raw={"choices": [{"finish_reason": "stop"}]},
        )
        with patch(PATCH_SHOULD_SEND, return_value=True):
            set_llm_chat_response(event, span)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == ["stop"]

    def test_finish_reasons_span_attr_not_gated_by_should_send_prompts(self):
        """gen_ai.response.finish_reasons is metadata, not content — never gated."""
        span = _recording_span()
        msg = _chat_message("assistant", "Done.")
        event = MagicMock()
        event.response = MagicMock(
            message=msg,
            raw={"choices": [{"finish_reason": "stop"}]},
        )
        with patch(PATCH_SHOULD_SEND, return_value=False):
            set_llm_chat_response(event, span)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == ["stop"]


# ===========================================================================
# set_llm_chat_response_model_attributes — finish_reasons, tokens, model
# ===========================================================================

class TestSetLlmChatResponseModelAttributes:
    def _event_with_raw(self, raw):
        event = MagicMock()
        event.response = MagicMock(raw=raw)
        return event

    def test_sets_model(self):
        span = _recording_span()
        raw = SimpleNamespace(model="gpt-4o")
        set_llm_chat_response_model_attributes(self._event_with_raw(raw), span)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL) == "gpt-4o"

    def test_sets_token_usage_openai(self):
        span = _recording_span()
        raw = SimpleNamespace(
            model="gpt-4",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        set_llm_chat_response_model_attributes(self._event_with_raw(raw), span)
        assert _attr(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 10
        assert _attr(span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 20
        assert _attr(span, SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS) == 30

    def test_no_legacy_total_tokens_attr(self):
        span = _recording_span()
        raw = SimpleNamespace(
            model="gpt-4",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        set_llm_chat_response_model_attributes(self._event_with_raw(raw), span)
        assert not _has_attr(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS)

    def test_cohere_token_usage(self):
        span = _recording_span()
        raw = SimpleNamespace(
            model="command-r",
            meta=SimpleNamespace(tokens=SimpleNamespace(input_tokens=5, output_tokens=15)),
            finish_reason="COMPLETE",
        )
        set_llm_chat_response_model_attributes(self._event_with_raw(raw), span)
        assert _attr(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 5
        assert _attr(span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 15
        assert _attr(span, SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS) == 20

    def test_response_id_set(self):
        span = _recording_span()
        raw = SimpleNamespace(model="gpt-4", id="chatcmpl-abc123")
        set_llm_chat_response_model_attributes(self._event_with_raw(raw), span)
        assert _attr(span, GenAIAttributes.GEN_AI_RESPONSE_ID) == "chatcmpl-abc123"

    def test_none_raw_returns_early(self):
        span = _recording_span()
        event = MagicMock()
        event.response = MagicMock(raw=None)
        set_llm_chat_response_model_attributes(event, span)
        # Only is_recording check, no attributes set
        assert not _has_attr(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL)


# ===========================================================================
# set_llm_predict_response — completion output messages
# ===========================================================================

class TestSetLlmPredictResponse:
    def test_sets_output_messages_json(self):
        span = _recording_span()
        event = MagicMock()
        event.output = "The answer is 42."
        with patch(PATCH_SHOULD_SEND, return_value=True):
            set_llm_predict_response(event, span)
        raw_out = _attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        assert raw_out is not None
        msgs = json.loads(raw_out)
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["parts"][0]["content"] == "The answer is 42."

    def test_gated_by_should_send_prompts(self):
        span = _recording_span()
        event = MagicMock()
        event.output = "text"
        with patch(PATCH_SHOULD_SEND, return_value=False):
            set_llm_predict_response(event, span)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)

    def test_none_output(self):
        span = _recording_span()
        event = MagicMock()
        event.output = None
        with patch(PATCH_SHOULD_SEND, return_value=True):
            set_llm_predict_response(event, span)
        raw_out = _attr(span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        msgs = json.loads(raw_out)
        assert msgs[0]["parts"] == []

    def test_no_legacy_indexed_attributes(self):
        span = _recording_span()
        event = MagicMock()
        event.output = "Reply"
        with patch(PATCH_SHOULD_SEND, return_value=True):
            set_llm_predict_response(event, span)
        for call in span.set_attribute.call_args_list:
            assert not call.args[0].startswith(f"{GenAIAttributes.GEN_AI_COMPLETION}.")


# ===========================================================================
# set_embedding — semconv attributes
# ===========================================================================

class TestSetEmbedding:
    def test_sets_operation_name_embeddings(self):
        span = _recording_span()
        event = MagicMock()
        event.model_dict = {"model_name": "text-embedding-3-small"}
        set_embedding(event, span)
        assert _attr(span, GenAIAttributes.GEN_AI_OPERATION_NAME) == "embeddings"

    def test_sets_request_model(self):
        span = _recording_span()
        event = MagicMock()
        event.model_dict = {"model_name": "text-embedding-3-small"}
        set_embedding(event, span)
        assert _attr(span, GenAIAttributes.GEN_AI_REQUEST_MODEL) == "text-embedding-3-small"

    def test_no_legacy_embedding_model_name(self):
        """Must NOT emit legacy 'embedding.model_name' attribute."""
        span = _recording_span()
        event = MagicMock()
        event.model_dict = {"model_name": "text-embedding-3-small"}
        set_embedding(event, span)
        assert not _has_attr(span, "embedding.model_name")


# ===========================================================================
# set_rerank / set_rerank_model_attributes — semconv attributes
# ===========================================================================

class TestSetRerankModelAttributes:
    def test_sets_operation_name(self):
        span = _recording_span()
        event = MagicMock()
        event.model_name = "rerank-v3.5"
        event.top_n = 5
        set_rerank_model_attributes(event, span)
        assert _attr(span, GenAIAttributes.GEN_AI_OPERATION_NAME) == "rerank"

    def test_sets_request_model(self):
        span = _recording_span()
        event = MagicMock()
        event.model_name = "rerank-v3.5"
        event.top_n = 5
        set_rerank_model_attributes(event, span)
        assert _attr(span, GenAIAttributes.GEN_AI_REQUEST_MODEL) == "rerank-v3.5"

    def test_sets_top_n(self):
        """top_n is a rerank-specific param — kept as rerank.top_n (no semconv equivalent)."""
        span = _recording_span()
        event = MagicMock()
        event.model_name = "rerank-v3.5"
        event.top_n = 3
        set_rerank_model_attributes(event, span)
        assert _attr(span, "rerank.top_n") == 3

    def test_no_legacy_rerank_model_name(self):
        """Must NOT emit legacy 'rerank.model_name' attribute."""
        span = _recording_span()
        event = MagicMock()
        event.model_name = "rerank-v3.5"
        event.top_n = 5
        set_rerank_model_attributes(event, span)
        assert not _has_attr(span, "rerank.model_name")


class TestSetRerank:
    def test_sets_input_messages_with_query(self):
        span = _recording_span()
        event = MagicMock()
        event.query.query_str = "what is the meaning of life?"
        with patch(PATCH_SHOULD_SEND, return_value=True):
            set_rerank(event, span)
        raw = _attr(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES)
        assert raw is not None
        msgs = json.loads(raw)
        assert msgs[0]["role"] == "user"
        assert msgs[0]["parts"][0]["content"] == "what is the meaning of life?"

    def test_gated_by_should_send_prompts(self):
        span = _recording_span()
        event = MagicMock()
        event.query.query_str = "query"
        with patch(PATCH_SHOULD_SEND, return_value=False):
            set_rerank(event, span)
        assert not _has_attr(span, GenAIAttributes.GEN_AI_INPUT_MESSAGES)

    def test_no_legacy_rerank_query(self):
        """Must NOT emit legacy 'rerank.query' attribute."""
        span = _recording_span()
        event = MagicMock()
        event.query.query_str = "query"
        with patch(PATCH_SHOULD_SEND, return_value=True):
            set_rerank(event, span)
        assert not _has_attr(span, "rerank.query")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
