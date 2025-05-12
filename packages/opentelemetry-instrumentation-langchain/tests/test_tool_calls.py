import json
import pytest

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from opentelemetry.semconv_ai import SpanAttributes


def food_analysis(name: str, healthy: bool, calories: int, taste_profile: list[str]) -> str:
    return "pass"


@pytest.mark.vcr
def test_tool_calls(exporter):
    query_text = "Analyze the following food item: avocado"
    query = [HumanMessage(content=query_text)]
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    model_with_tools = model.bind_tools([food_analysis])
    result = model_with_tools.invoke(query)
    spans = exporter.get_finished_spans()

    span_names = set(span.name for span in spans)
    expected_spans = {"ChatOpenAI.chat"}
    assert expected_spans.issubset(span_names)

    chat_span = next(
        span for span in spans if span.name == "ChatOpenAI.chat"
    )

    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == query_text
    assert chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name"] == "food_analysis"

    arguments = chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments"]
    assert (
        arguments == result.model_dump().get("additional_kwargs").get("tool_calls")[0].get("function").get("arguments")
    )


@pytest.mark.vcr
def test_tool_calls_with_history(exporter):
    def get_weather(location: str) -> str:
        return "sunny"

    messages: list[BaseMessage] = [
        SystemMessage(content="Be crisp and friendly."),
        HumanMessage(content="Hey, what's the weather in San Francisco?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"location": "San Francisco"},
                    "id": "tool_123",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="Sunny as always!", tool_call_id="tool_123"),
        HumanMessage(content="What's the weather in London?"),
    ]
    model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    model_with_tools = model.bind_tools([get_weather])
    result = model_with_tools.invoke(messages)
    spans = exporter.get_finished_spans()

    span_names = set(span.name for span in spans)
    expected_spans = {"ChatOpenAI.chat"}
    assert expected_spans.issubset(span_names)

    chat_span = next(
        span for span in spans if span.name == "ChatOpenAI.chat"
    )

    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == messages[0].content
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"

    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == messages[1].content
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "user"

    assert json.loads(chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"]) == messages[2].tool_calls
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.role"] == "assistant"

    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.3.content"] == messages[3].content
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.3.role"] == "tool"

    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.4.content"] == messages[4].content
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.4.role"] == "user"

    assert chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name"] == "get_weather"

    arguments = chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments"]
    assert (
        arguments == result.model_dump().get("additional_kwargs").get("tool_calls")[0].get("function").get("arguments")
    )
