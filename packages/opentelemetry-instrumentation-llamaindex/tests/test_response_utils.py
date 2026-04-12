"""Unit tests for _response_utils — response extraction utilities."""

from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.llamaindex._response_utils import (
    TokenUsage,
    detect_provider_name,
    extract_finish_reasons,
    extract_model_from_raw,
    extract_response_id,
    extract_token_usage,
)


# ===========================================================================
# detect_provider_name
# ===========================================================================

class TestDetectProviderName:
    def test_openai_class(self):
        inst = type("OpenAI", (), {})()
        assert detect_provider_name(inst) == "openai"

    def test_cohere_class(self):
        inst = type("Cohere", (), {})()
        assert detect_provider_name(inst) == "cohere"

    def test_anthropic_class(self):
        inst = type("Anthropic", (), {})()
        assert detect_provider_name(inst) == "anthropic"

    def test_groq_class(self):
        inst = type("Groq", (), {})()
        assert detect_provider_name(inst) == "groq"

    def test_mistralai_class(self):
        inst = type("MistralAI", (), {})()
        assert detect_provider_name(inst) == "mistral_ai"

    def test_bedrock_class(self):
        inst = type("Bedrock", (), {})()
        assert detect_provider_name(inst) == "aws.bedrock"

    def test_gemini_class(self):
        inst = type("Gemini", (), {})()
        assert detect_provider_name(inst) == "gcp.gemini"

    def test_ollama_class(self):
        inst = type("Ollama", (), {})()
        assert detect_provider_name(inst) == "ollama"

    def test_custom_llm_class(self):
        inst = type("MyCustomLLM", (), {})()
        assert detect_provider_name(inst) == "mycustomllm"

    def test_none_instance(self):
        assert detect_provider_name(None) is None

    def test_from_string_class_name(self):
        assert detect_provider_name("OpenAI") == "openai"

    def test_from_string_unknown(self):
        assert detect_provider_name("SomeProvider") == "someprovider"

    def test_azure_openai(self):
        assert detect_provider_name("AzureOpenAI") == "azure.ai.openai"

    def test_deepseek(self):
        assert detect_provider_name("DeepSeek") == "deepseek"


# ===========================================================================
# extract_model_from_raw
# ===========================================================================

class TestExtractModelFromRaw:
    def test_object_with_model_attr(self):
        raw = SimpleNamespace(model="gpt-4")
        assert extract_model_from_raw(raw) == "gpt-4"

    def test_dict_with_model_key(self):
        assert extract_model_from_raw({"model": "gpt-4"}) == "gpt-4"

    def test_no_model_returns_none(self):
        assert extract_model_from_raw(SimpleNamespace()) is None

    def test_none_in_dict(self):
        assert extract_model_from_raw({"model": None}) is None


# ===========================================================================
# extract_response_id
# ===========================================================================

class TestExtractResponseId:
    def test_object_with_id_attr(self):
        raw = SimpleNamespace(id="chatcmpl-abc123")
        assert extract_response_id(raw) == "chatcmpl-abc123"

    def test_dict_with_id_key(self):
        assert extract_response_id({"id": "resp-1"}) == "resp-1"

    def test_no_id_returns_none(self):
        assert extract_response_id({}) is None


# ===========================================================================
# extract_token_usage
# ===========================================================================

class TestExtractTokenUsage:
    def test_openai_format_object(self):
        raw = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )
        result = extract_token_usage(raw)
        assert result == TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)

    def test_openai_format_dict(self):
        raw = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }
        result = extract_token_usage(raw)
        assert result == TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)

    def test_cohere_meta_tokens_format(self):
        raw = SimpleNamespace(
            meta=SimpleNamespace(tokens=SimpleNamespace(input_tokens=5, output_tokens=15))
        )
        result = extract_token_usage(raw)
        assert result.input_tokens == 5
        assert result.output_tokens == 15
        assert result.total_tokens == 20

    def test_cohere_meta_tokens_dict(self):
        raw = {"meta": {"tokens": {"input_tokens": 5, "output_tokens": 15}}}
        result = extract_token_usage(raw)
        assert result.input_tokens == 5
        assert result.output_tokens == 15

    def test_cohere_meta_billed_units_format(self):
        raw = SimpleNamespace(
            meta=SimpleNamespace(
                tokens=None,
                billed_units=SimpleNamespace(input_tokens=3, output_tokens=7),
            )
        )
        result = extract_token_usage(raw)
        assert result.input_tokens == 3
        assert result.output_tokens == 7
        assert result.total_tokens == 10

    def test_cohere_meta_billed_units_dict(self):
        raw = {"meta": {"billed_units": {"input_tokens": 3, "output_tokens": 7}}}
        result = extract_token_usage(raw)
        assert result.input_tokens == 3
        assert result.output_tokens == 7

    def test_no_usage_returns_empty(self):
        result = extract_token_usage(SimpleNamespace())
        assert result == TokenUsage()

    def test_partial_usage(self):
        raw = {"usage": {"prompt_tokens": 10}}
        result = extract_token_usage(raw)
        assert result.input_tokens == 10
        assert result.output_tokens is None


# ===========================================================================
# extract_finish_reasons
# ===========================================================================

class TestExtractFinishReasons:
    def test_openai_choices_object(self):
        choice = SimpleNamespace(finish_reason="stop")
        raw = SimpleNamespace(choices=[choice])
        assert extract_finish_reasons(raw) == ["stop"]

    def test_openai_choices_dict(self):
        raw = {"choices": [{"finish_reason": "stop"}]}
        assert extract_finish_reasons(raw) == ["stop"]

    def test_openai_tool_calls_passed_through(self):
        raw = {"choices": [{"finish_reason": "tool_calls"}]}
        assert extract_finish_reasons(raw) == ["tool_calls"]

    def test_multiple_choices(self):
        raw = {"choices": [{"finish_reason": "stop"}, {"finish_reason": "length"}]}
        assert extract_finish_reasons(raw) == ["stop", "length"]

    def test_anthropic_stop_reason(self):
        raw = SimpleNamespace(stop_reason="end_turn")
        assert extract_finish_reasons(raw) == ["stop"]

    def test_anthropic_stop_reason_dict(self):
        raw = {"stop_reason": "end_turn"}
        assert extract_finish_reasons(raw) == ["stop"]

    def test_cohere_finish_reason(self):
        raw = SimpleNamespace(finish_reason="COMPLETE")
        assert extract_finish_reasons(raw) == ["stop"]

    def test_cohere_finish_reason_dict(self):
        raw = {"finish_reason": "MAX_TOKENS"}
        assert extract_finish_reasons(raw) == ["length"]

    def test_none_raw(self):
        assert extract_finish_reasons(None) == []

    def test_no_finish_reason(self):
        assert extract_finish_reasons(SimpleNamespace()) == []

    def test_none_finish_reason_in_choices(self):
        raw = {"choices": [{"finish_reason": None}]}
        assert extract_finish_reasons(raw) == []

    def test_empty_choices(self):
        raw = {"choices": []}
        assert extract_finish_reasons(raw) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
