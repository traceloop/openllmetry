import pytest
from unittest.mock import Mock

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry.instrumentation.langchain.span_utils import set_chat_response_usage
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.semconv_ai import SpanAttributes


def _mock_span():
    span = Mock()
    span.is_recording.return_value = True
    span.attributes = {}

    def set_attribute(key, value):
        span.attributes[key] = value

    span.set_attribute = set_attribute
    return span


@pytest.mark.parametrize(
    "response_metadata",
    [
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 16,
                "total_tokens": 26,
            }
        },
        {
            "prompt_tokens": 10,
            "completion_tokens": 16,
            "total_tokens": 26,
        },
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 16,
            }
        },
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 16,
                "total_token_count": 26,
            }
        },
    ],
)
def test_chat_response_usage_reads_databricks_response_metadata(response_metadata):
    span = _mock_span()
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="Hello!",
                        response_metadata=response_metadata,
                    )
                )
            ]
        ]
    )

    set_chat_response_usage(
        span,
        response,
        token_histogram=Mock(),
        record_token_usage=False,
        model_name="databricks-claude-sonnet",
    )

    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 16
    assert span.attributes[SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS] == 26
