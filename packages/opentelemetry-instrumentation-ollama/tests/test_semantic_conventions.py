from unittest.mock import MagicMock
from opentelemetry.instrumentation.ollama.span_utils import (
    set_model_input_attributes,
    set_model_response_attributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes


def test_set_model_input_attributes_semantic_conventions():
    span = MagicMock()
    span.is_recording.return_value = True

    kwargs = {
        "json": {
            "model": "llama3",
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "num_predict": 128,
                "stop": ["\n", "User:"],
            },
        },
        "stream": True,
    }

    set_model_input_attributes(span, kwargs)

    # Verify standard model attributes
    span.set_attribute.assert_any_call(GenAIAttributes.GEN_AI_REQUEST_MODEL, "llama3")
    span.set_attribute.assert_any_call(SpanAttributes.LLM_IS_STREAMING, True)
    span.set_attribute.assert_any_call(SpanAttributes.GEN_AI_IS_STREAMING, True)

    # Verify request options
    span.set_attribute.assert_any_call(GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE, 0.7)
    span.set_attribute.assert_any_call(GenAIAttributes.GEN_AI_REQUEST_TOP_P, 0.9)
    span.set_attribute.assert_any_call(GenAIAttributes.GEN_AI_REQUEST_TOP_K, 40)
    span.set_attribute.assert_any_call(SpanAttributes.GEN_AI_REQUEST_REPETITION_PENALTY, 1.1)
    span.set_attribute.assert_any_call(GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY, 0.0)
    span.set_attribute.assert_any_call(GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY, 0.0)
    span.set_attribute.assert_any_call(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS, 128)
    span.set_attribute.assert_any_call(GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES, ("\n", "User:"))


def test_set_model_response_attributes_semantic_conventions():
    span = MagicMock()
    span.is_recording.return_value = True

    response = {
        "model": "llama3",
        "prompt_eval_count": 10,
        "eval_count": 20,
        "done_reason": "stop",
    }

    set_model_response_attributes(span, None, LLMRequestTypeValues.CHAT, response)

    span.set_attribute.assert_any_call(GenAIAttributes.GEN_AI_RESPONSE_MODEL, "llama3")
    span.set_attribute.assert_any_call(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, 10)
    span.set_attribute.assert_any_call(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, 20)
    span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, 30)

    # Verify finish reasons
    span.set_attribute.assert_any_call(SpanAttributes.GEN_AI_RESPONSE_FINISH_REASON, "stop")
    span.set_attribute.assert_any_call(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, ("stop",))
