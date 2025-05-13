import json
import pytest

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
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
    result_arguments = result.model_dump().get(
        "additional_kwargs"
    ).get("tool_calls")[0].get("function").get("arguments")
    assert (
        json.loads(arguments) == json.loads(result_arguments)
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

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatOpenAI.chat"

    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == messages[0].content
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"

    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"] == messages[1].content
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "user"

    assert (
        chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.tool_calls.0.name"]
        == messages[2].tool_calls[0]["name"]
    )
    assert (
        json.loads(chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.tool_calls.0.arguments"])
        == messages[2].tool_calls[0]["args"]
    )
    assert (
        chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.tool_calls.0.id"]
        == messages[2].tool_calls[0]["id"]
    )
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.role"] == "assistant"

    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.3.content"] == messages[3].content
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.3.role"] == "tool"

    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.4.content"] == messages[4].content
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.4.role"] == "user"

    assert chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name"] == "get_weather"

    arguments = chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments"]
    result_arguments = result\
        .model_dump()["additional_kwargs"]["tool_calls"][0]["function"]["arguments"]
    assert (
        json.loads(arguments) == json.loads(result_arguments)
    )


@pytest.mark.vcr
def test_tool_calls_anthropic_text_block(exporter):
    # This test checks for cases when anthropic prepends a tool call with a text block.

    def get_weather(location: str) -> str:
        return "sunny"

    def get_news(location: str) -> str:
        return "Not much"

    messages: list[BaseMessage] = [
        HumanMessage(content="Hey, what's the weather in San Francisco? Also, any news in town?"),
    ]
    model = ChatAnthropic(model="claude-3-5-haiku-latest")
    model_with_tools = model.bind_tools([get_weather, get_news])
    result = model_with_tools.invoke(messages)
    spans = exporter.get_finished_spans()

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatAnthropic.chat"
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == messages[0].content
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"

    assert chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    # Test that we write both the content and the tool calls
    assert chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == result.content[0]["text"]
    assert (
        chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.id"]
        == "toolu_016q9vtSd8CY2vnZSpEp1j4o"
    )
    assert (
        chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name"]
        == "get_weather"
    )
    assert (
        json.loads(chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments"])
        == {"location": "San Francisco"}
    )


@pytest.mark.vcr
def test_tool_calls_anthropic_text_block_and_history(exporter):
    # This test checks for cases when anthropic prepends a tool call with a text block
    # and then the response messaged is added to the history.
    def get_weather(location: str) -> str:
        return "sunny"

    def get_news(location: str) -> str:
        return "Not much"

    messages: list[BaseMessage] = [
        HumanMessage(content="Hey, what's the weather in San Francisco? Also, any news in town?"),
        AIMessage(
            content=[
                {
                    'text': "I'll help you with that by checking the weather and news"
                    " for San Francisco right away.\n\nFirst, let's check the weather:",
                    'type': 'text'
                },
                {
                    'id': 'toolu_016q9vtSd8CY2vnZSpEp1j4o',
                    'input': {'location': 'San Francisco'},
                    'name': 'get_weather',
                    'type': 'tool_use',
                }
            ],
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"location": "San Francisco"},
                    "id": "toolu_016q9vtSd8CY2vnZSpEp1j4o",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="Sunny as always!", tool_call_id="toolu_016q9vtSd8CY2vnZSpEp1j4o"),
    ]
    model = ChatAnthropic(model="claude-3-5-haiku-latest")
    model_with_tools = model.bind_tools([get_weather, get_news])
    result = model_with_tools.invoke(messages)
    spans = exporter.get_finished_spans()

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatAnthropic.chat"
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == messages[0].content
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "assistant"

    assert (
        chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls.0.name"]
        == messages[1].tool_calls[0]["name"]
    )
    assert (
        json.loads(chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls.0.arguments"])
        == messages[1].tool_calls[0]["args"]
    )
    assert (
        chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.tool_calls.0.id"]
        == messages[1].tool_calls[0]["id"]
    )

    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.role"] == "tool"
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.2.content"] == messages[2].content

    assert chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"
    # Test that we write both the content and the tool calls
    assert (
        chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == result.content[0]["text"]
    )
    assert (
        chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.id"]
        == "toolu_012guEZNJ5yH5jxHKWAkzCzh"
    )
    assert (
        chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name"]
        == "get_news"
    )
    assert (
        json.loads(chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments"])
        == {"location": "San Francisco"}
    )


@pytest.mark.vcr
def test_parallel_tool_calls(exporter):
    def get_weather(location: str) -> str:
        return "sunny"

    def get_news(location: str) -> str:
        return "Not much"

    messages: list[BaseMessage] = [
        HumanMessage(content="Hey, what's the weather in San Francisco? Also, any news in town?"),
    ]
    model = ChatOpenAI(model="gpt-4.1-nano")
    model_with_tools = model.bind_tools([get_weather, get_news])
    model_with_tools.invoke(messages)
    spans = exporter.get_finished_spans()

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatOpenAI.chat"
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == messages[0].content
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "user"

    assert chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"] == "assistant"

    assert (
        chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.id"]
        == "call_EgULHWKqGjuB36aUeiOSpALZ"
    )
    assert (
        chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name"]
        == "get_weather"
    )
    assert (
        json.loads(chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments"])
        == {"location": "San Francisco"}
    )
    assert (
        chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.id"]
        == "call_Xer9QGOTDMG2Bxn9AKGiVM14"
    )
    assert (
        chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.name"]
        == "get_news"
    )
    assert (
        json.loads(chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.1.arguments"])
        == {"location": "San Francisco"}
    )
