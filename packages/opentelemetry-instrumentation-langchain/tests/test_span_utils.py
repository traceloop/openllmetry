import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from opentelemetry.instrumentation.langchain.span_utils import set_chat_response
from opentelemetry.instrumentation.langchain.utils import TRACELOOP_TRACE_CONTENT
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


class _DummySpan:
    def __init__(self):
        self.attributes = {}

    def is_recording(self) -> bool:
        return True

    def set_attribute(self, key, value) -> None:
        self.attributes[key] = value


@pytest.fixture(autouse=True)
def _enable_prompt_content(monkeypatch):
    monkeypatch.setenv(TRACELOOP_TRACE_CONTENT, "true")


def _make_result(message: AIMessage) -> tuple[_DummySpan, LLMResult]:
    span = _DummySpan()
    generation = ChatGeneration(message=message)
    result = LLMResult(generations=[[generation]])
    return span, result


@pytest.mark.parametrize(
    "message",
    [
        AIMessage(content="hi"),
        AIMessage(
            content="tool reply",
            additional_kwargs={
                "function_call": {"name": "call_weather", "arguments": "{}"}
            },
        ),
        AIMessage(
            content="another reply",
            tool_calls=[{"name": "foo", "args": {"city": "SF"}, "id": "1"}],
        ),
    ],
)
def test_chat_generation_role_is_assistant(message):
    span, result = _make_result(message)

    set_chat_response(span, result)

    assert (
        span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"]
        == "assistant"
    )

    tool_call = message.additional_kwargs.get("function_call")
    if tool_call:
        prefix = f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0"
        assert span.attributes[prefix + ".name"] == tool_call["name"]
        assert span.attributes[prefix + ".arguments"] == tool_call["arguments"]

    if message.tool_calls:
        prefix = f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0"
        assert span.attributes[prefix + ".name"] == message.tool_calls[0]["name"]
        assert span.attributes[prefix + ".id"] == message.tool_calls[0]["id"]
        recorded_args = json.loads(span.attributes[prefix + ".arguments"])
        assert recorded_args == message.tool_calls[0]["args"]


def test_plain_generation_defaults_to_assistant_role():
    span = _DummySpan()
    generation = Generation(text="plain completion")
    result = LLMResult(generations=[[generation]])

    set_chat_response(span, result)

    prefix = f"{GenAIAttributes.GEN_AI_COMPLETION}.0"
    assert prefix + ".role" not in span.attributes
    assert span.attributes[prefix + ".content"] == "plain completion"
