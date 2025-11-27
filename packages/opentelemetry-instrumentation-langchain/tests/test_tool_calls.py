import json
from typing import List

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


def food_analysis(
    name: str, healthy: bool, calories: int, taste_profile: List[str]
) -> str:
    return "pass"


@pytest.mark.vcr
def test_tool_calls(instrument_legacy, span_exporter, log_exporter):
    query_text = "Analyze the following food item: avocado"
    query = [HumanMessage(content=query_text)]
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    model_with_tools = model.bind_tools([food_analysis])
    _ = model_with_tools.invoke(query)
    # spans = span_exporter.get_finished_spans()

    # span_names = set(span.name for span in spans)
    # expected_spans = {"ChatOpenAI.chat"}
    # assert expected_spans.issubset(span_names)

    # chat_span = next(
    #     span for span in spans if span.name == "ChatOpenAI.chat"
    # )

    # assert chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name"] == "food_analysis"
    # assert json.loads(chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.parameters"]) == {
    #     "properties": {
    #         "name": {"type": "string"},
    #         "healthy": {"type": "boolean"},
    #         "calories": {"type": "integer"},
    #         "taste_profile": {"type": "array", "items": {"type": "string"}},
    #     },
    #     "required": ["name", "healthy", "calories", "taste_profile"],
    #     "type": "object",
    # }

    # assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"] == query_text
    # assert (
    #     chat_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.name"]
    #     == "food_analysis"
    # )

    # arguments = chat_span.attributes[
    #     f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.arguments"
    # ]
    # assert json.loads(arguments) == json.loads(
    #     result.model_dump()
    #     .get("additional_kwargs")
    #     .get("tool_calls")[0]
    #     .get("function")
    #     .get("arguments")
    # )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.vcr
def test_tool_calls_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    query_text = "Analyze the following food item: avocado"
    query = [HumanMessage(content=query_text)]
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    model_with_tools = model.bind_tools([food_analysis])
    model_with_tools.invoke(query)
    spans = span_exporter.get_finished_spans()

    span_names = set(span.name for span in spans)
    expected_spans = {"ChatOpenAI.chat"}
    assert expected_spans.issubset(span_names)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {"content": query_text})

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {"content": ""},
        "tool_calls": [
            {
                "function": {
                    "arguments": {
                        "calories": 240,
                        "healthy": True,
                        "name": "avocado",
                        "taste_profile": ["creamy", "buttery", "nutty"],
                    },
                    "name": "food_analysis",
                },
                "id": "call_eZXHC28rALvooYh4VGZgVQ9t",
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.vcr
def test_tool_calls_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    query_text = "Analyze the following food item: avocado"
    query = [HumanMessage(content=query_text)]
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    model_with_tools = model.bind_tools([food_analysis])
    model_with_tools.invoke(query)
    spans = span_exporter.get_finished_spans()

    span_names = set(span.name for span in spans)
    expected_spans = {"ChatOpenAI.chat"}
    assert expected_spans.issubset(span_names)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {},
        "tool_calls": [
            {
                "function": {"name": "food_analysis"},
                "id": "call_eZXHC28rALvooYh4VGZgVQ9t",
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.vcr
def test_tool_calls_with_history(instrument_legacy, span_exporter, log_exporter):
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
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatOpenAI.chat"

    assert chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name"] == "get_weather"
    assert json.loads(chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.parameters"]) == {
        "properties": {
            "location": {"type": "string"},
        },
        "required": ["location"],
        "type": "object",
    }

    assert chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name"] == "get_weather"
    assert json.loads(chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.parameters"]) == {
        "properties": {
            "location": {"type": "string"},
        },
        "required": ["location"],
        "type": "object",
    }

    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == messages[0].content
    )
    assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "system"

    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"]
        == messages[1].content
    )
    assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"] == "user"

    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.tool_calls.0.name"]
        == messages[2].tool_calls[0]["name"]
    )
    assert (
        json.loads(
            chat_span.attributes[
                f"{GenAIAttributes.GEN_AI_PROMPT}.2.tool_calls.0.arguments"
            ]
        )
        == messages[2].tool_calls[0]["args"]
    )
    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.tool_calls.0.id"]
        == messages[2].tool_calls[0]["id"]
    )
    assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.role"] == "assistant"

    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.3.content"]
        == messages[3].content
    )
    assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.3.role"] == "tool"
    assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.3.tool_call_id"] == messages[3].tool_call_id

    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.4.content"]
        == messages[4].content
    )
    assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.4.role"] == "user"

    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.name"]
        == "get_weather"
    )

    arguments = chat_span.attributes[
        f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.arguments"
    ]
    result_arguments = result.model_dump()["additional_kwargs"]["tool_calls"][0][
        "function"
    ]["arguments"]
    assert json.loads(arguments) == json.loads(result_arguments)

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.vcr
def test_tool_calls_with_history_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
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
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatOpenAI.chat"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 6

    # Validate system message Event
    assert_message_in_logs(
        logs[0], "gen_ai.system.message", {"content": "Be crisp and friendly."}
    )

    # Validate user message Event
    assert_message_in_logs(
        logs[1],
        "gen_ai.user.message",
        {"content": "Hey, what's the weather in San Francisco?"},
    )

    # Validate AI message Event
    assert_message_in_logs(
        logs[2],
        "gen_ai.assistant.message",
        {
            "content": "",
            "tool_calls": [
                {
                    "id": messages[2].tool_calls[0]["id"],
                    "function": {
                        "name": messages[2].tool_calls[0]["name"],
                        "arguments": messages[2].tool_calls[0]["args"],
                    },
                    "type": "function",
                }
            ],
        },
    )

    # Validate tool message Event
    assert_message_in_logs(
        logs[3], "gen_ai.tool.message", {"content": "Sunny as always!"}
    )

    # Validate second user message Event
    assert_message_in_logs(
        logs[4], "gen_ai.user.message", {"content": "What's the weather in London?"}
    )

    # Validate AI choice Event
    tool_call = result.model_dump()["additional_kwargs"]["tool_calls"][0]
    choice_event = {
        "index": 0,
        "message": {"content": ""},
        "finish_reason": "tool_calls",
        "tool_calls": [
            {
                "id": tool_call["id"],
                "function": {
                    "name": tool_call["function"]["name"],
                    "arguments": json.loads(tool_call["function"]["arguments"]),
                },
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[5], "gen_ai.choice", choice_event)


@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.vcr
def test_tool_calls_with_history_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
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
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatOpenAI.chat"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 6

    # Validate system message Event
    assert_message_in_logs(logs[0], "gen_ai.system.message", {})

    # Validate user message Event
    assert_message_in_logs(logs[1], "gen_ai.user.message", {})

    # Validate AI message Event
    assert_message_in_logs(
        logs[2],
        "gen_ai.assistant.message",
        {
            "tool_calls": [
                {
                    "id": messages[2].tool_calls[0]["id"],
                    "function": {"name": "get_weather"},
                    "type": "function",
                }
            ],
        },
    )

    # Validate tool message Event
    assert_message_in_logs(logs[3], "gen_ai.tool.message", {})

    # Validate second user message Event
    assert_message_in_logs(logs[4], "gen_ai.user.message", {})

    # Validate AI choice Event
    tool_call = result.model_dump()["additional_kwargs"]["tool_calls"][0]
    choice_event = {
        "index": 0,
        "message": {},
        "finish_reason": "tool_calls",
        "tool_calls": [
            {
                "id": tool_call["id"],
                "function": {"name": "get_weather"},
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[5], "gen_ai.choice", choice_event)


@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.vcr
def test_tool_calls_anthropic_text_block(
    instrument_legacy, span_exporter, log_exporter
):
    # This test checks for cases when anthropic prepends a tool call with a text block.

    def get_weather(location: str) -> str:
        return "sunny"

    def get_news(location: str) -> str:
        return "Not much"

    messages: list[BaseMessage] = [
        HumanMessage(
            content="Hey, what's the weather in San Francisco? Also, any news in town?"
        ),
    ]
    model = ChatAnthropic(model="claude-3-5-haiku-latest")
    model_with_tools = model.bind_tools([get_weather, get_news])
    result = model_with_tools.invoke(messages)
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatAnthropic.chat"

    assert chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name"] == "get_weather"
    assert json.loads(chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.parameters"]) == {
        "properties": {
            "location": {"type": "string"},
        },
        "required": ["location"],
        "type": "object",
    }

    assert chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.name"] == "get_news"
    assert json.loads(chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.parameters"]) == {
        "properties": {
            "location": {"type": "string"},
        },
        "required": ["location"],
        "type": "object",
    }

    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == messages[0].content
    )
    assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "user"

    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"] == "assistant"
    )
    # Test that we write both the content and the tool calls
    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == result.content[0]["text"]
    )
    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.id"]
        == "toolu_016q9vtSd8CY2vnZSpEp1j4o"
    )
    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.name"]
        == "get_weather"
    )
    assert json.loads(
        chat_span.attributes[
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.arguments"
        ]
    ) == {"location": "San Francisco"}

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.vcr
def test_tool_calls_anthropic_text_block_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    # This test checks for cases when anthropic prepends a tool call with a text block.

    def get_weather(location: str) -> str:
        return "sunny"

    def get_news(location: str) -> str:
        return "Not much"

    messages: list[BaseMessage] = [
        HumanMessage(
            content="Hey, what's the weather in San Francisco? Also, any news in town?"
        ),
    ]
    model = ChatAnthropic(model="claude-3-5-haiku-latest")
    model_with_tools = model.bind_tools([get_weather, get_news])
    result = model_with_tools.invoke(messages)
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatAnthropic.chat"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {
            "content": "Hey, what's the weather in San Francisco? Also, any news in town?"
        },
    )

    # Validate AI choice Event
    result_dict = result.model_dump()
    choice_event = {
        "index": 0,
        "message": {"content": result_dict["content"][0]["text"]},
        "finish_reason": "unknown",
        "tool_calls": [
            {
                "id": result_dict["content"][1]["id"],
                "function": {
                    "name": result_dict["content"][1]["name"],
                    "arguments": result_dict["content"][1]["input"],
                },
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.vcr
def test_tool_calls_anthropic_text_block_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    # This test checks for cases when anthropic prepends a tool call with a text block.

    def get_weather(location: str) -> str:
        return "sunny"

    def get_news(location: str) -> str:
        return "Not much"

    messages: list[BaseMessage] = [
        HumanMessage(
            content="Hey, what's the weather in San Francisco? Also, any news in town?"
        ),
    ]
    model = ChatAnthropic(model="claude-3-5-haiku-latest")
    model_with_tools = model.bind_tools([get_weather, get_news])
    result = model_with_tools.invoke(messages)
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatAnthropic.chat"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate AI choice Event
    result_dict = result.model_dump()
    choice_event = {
        "index": 0,
        "message": {},
        "finish_reason": "unknown",
        "tool_calls": [
            {
                "id": result_dict["content"][1]["id"],
                "function": {"name": result_dict["content"][1]["name"]},
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.vcr
def test_tool_calls_anthropic_text_block_and_history(
    instrument_legacy, span_exporter, log_exporter
):
    # This test checks for cases when anthropic prepends a tool call with a text block
    # and then the response messaged is added to the history.
    def get_weather(location: str) -> str:
        return "sunny"

    def get_news(location: str) -> str:
        return "Not much"

    messages: list[BaseMessage] = [
        HumanMessage(
            content="Hey, what's the weather in San Francisco? Also, any news in town?"
        ),
        AIMessage(
            content=[
                {
                    "text": "I'll help you with that by checking the weather and news"
                    " for San Francisco right away.\n\nFirst, let's check the weather:",
                    "type": "text",
                },
                {
                    "id": "toolu_016q9vtSd8CY2vnZSpEp1j4o",
                    "input": {"location": "San Francisco"},
                    "name": "get_weather",
                    "type": "tool_use",
                },
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
        ToolMessage(
            content="Sunny as always!", tool_call_id="toolu_016q9vtSd8CY2vnZSpEp1j4o"
        ),
    ]
    model = ChatAnthropic(model="claude-3-5-haiku-latest")
    model_with_tools = model.bind_tools([get_weather, get_news])
    result = model_with_tools.invoke(messages)
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatAnthropic.chat"

    assert chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name"] == "get_weather"
    assert json.loads(chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.parameters"]) == {
        "properties": {
            "location": {"type": "string"},
        },
        "required": ["location"],
        "type": "object",
    }

    assert chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.name"] == "get_news"
    assert json.loads(chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.parameters"]) == {
        "properties": {
            "location": {"type": "string"},
        },
        "required": ["location"],
        "type": "object",
    }

    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == messages[0].content
    )
    assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "user"
    assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"] == "assistant"

    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.tool_calls.0.name"]
        == messages[1].tool_calls[0]["name"]
    )
    assert (
        json.loads(
            chat_span.attributes[
                f"{GenAIAttributes.GEN_AI_PROMPT}.1.tool_calls.0.arguments"
            ]
        )
        == messages[1].tool_calls[0]["args"]
    )
    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.tool_calls.0.id"]
        == messages[1].tool_calls[0]["id"]
    )

    assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.role"] == "tool"
    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.content"]
        == messages[2].content
    )
    assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.tool_call_id"] == messages[2].tool_call_id

    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"] == "assistant"
    )
    # Test that we write both the content and the tool calls
    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == result.content[0]["text"]
    )
    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.id"]
        == "toolu_012guEZNJ5yH5jxHKWAkzCzh"
    )
    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.name"]
        == "get_news"
    )
    assert json.loads(
        chat_span.attributes[
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.arguments"
        ]
    ) == {"location": "San Francisco"}

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.vcr
def test_tool_calls_anthropic_text_block_and_history_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    # This test checks for cases when anthropic prepends a tool call with a text block
    # and then the response messaged is added to the history.
    def get_weather(location: str) -> str:
        return "sunny"

    def get_news(location: str) -> str:
        return "Not much"

    messages: list[BaseMessage] = [
        HumanMessage(
            content="Hey, what's the weather in San Francisco? Also, any news in town?"
        ),
        AIMessage(
            content=[
                {
                    "text": "I'll help you with that by checking the weather and news"
                    " for San Francisco right away.\n\nFirst, let's check the weather:",
                    "type": "text",
                },
                {
                    "id": "toolu_016q9vtSd8CY2vnZSpEp1j4o",
                    "input": {"location": "San Francisco"},
                    "name": "get_weather",
                    "type": "tool_use",
                },
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
        ToolMessage(
            content="Sunny as always!", tool_call_id="toolu_016q9vtSd8CY2vnZSpEp1j4o"
        ),
    ]
    model = ChatAnthropic(model="claude-3-5-haiku-latest")
    model_with_tools = model.bind_tools([get_weather, get_news])
    result = model_with_tools.invoke(messages)
    print(result.model_dump())
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatAnthropic.chat"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    # Validate user message Event
    assert_message_in_logs(
        logs[0], "gen_ai.user.message", {"content": messages[0].content}
    )

    # Validate AI message Event
    assert_message_in_logs(
        logs[1],
        "gen_ai.assistant.message",
        {
            "content": messages[1].content,
            "tool_calls": [
                {
                    "id": messages[1].tool_calls[0]["id"],
                    "function": {
                        "name": messages[1].tool_calls[0]["name"],
                        "arguments": messages[1].tool_calls[0]["args"],
                    },
                    "type": "function",
                }
            ],
        },
    )

    # Validate tool message Event
    assert_message_in_logs(
        logs[2], "gen_ai.tool.message", {"content": messages[2].content}
    )

    # Validate AI choice Event
    result_dict = result.model_dump()
    choice_event = {
        "index": 0,
        "message": {"content": result_dict["content"][0]["text"]},
        "finish_reason": "unknown",
        "tool_calls": [
            {
                "id": result_dict["content"][1]["id"],
                "function": {
                    "name": result_dict["content"][1]["name"],
                    "arguments": result_dict["content"][1]["input"],
                },
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.vcr
def test_tool_message_with_tool_call_id(instrument_legacy, span_exporter, log_exporter):
    """Test that tool_call_id is properly set in span attributes for ToolMessage."""
    def sample_tool(query: str) -> str:
        return "Tool response"

    messages: list[BaseMessage] = [
        HumanMessage(content="Use the tool"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "sample_tool",
                    "args": {"query": "test"},
                    "id": "call_12345",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="Tool executed successfully", tool_call_id="call_12345"),
    ]

    model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    model_with_tools = model.bind_tools([sample_tool])
    model_with_tools.invoke(messages)
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatOpenAI.chat"

    # Verify that the tool_call_id is properly set for the ToolMessage
    assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.role"] == "tool"
    assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.content"] == "Tool executed successfully"
    assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.tool_call_id"] == "call_12345"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.vcr
def test_tool_calls_anthropic_text_block_and_history_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    # This test checks for cases when anthropic prepends a tool call with a text block
    # and then the response messaged is added to the history.
    def get_weather(location: str) -> str:
        return "sunny"

    def get_news(location: str) -> str:
        return "Not much"

    messages: list[BaseMessage] = [
        HumanMessage(
            content="Hey, what's the weather in San Francisco? Also, any news in town?"
        ),
        AIMessage(
            content=[
                {
                    "text": "I'll help you with that by checking the weather and news"
                    " for San Francisco right away.\n\nFirst, let's check the weather:",
                    "type": "text",
                },
                {
                    "id": "toolu_016q9vtSd8CY2vnZSpEp1j4o",
                    "input": {"location": "San Francisco"},
                    "name": "get_weather",
                    "type": "tool_use",
                },
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
        ToolMessage(
            content="Sunny as always!", tool_call_id="toolu_016q9vtSd8CY2vnZSpEp1j4o"
        ),
    ]
    model = ChatAnthropic(model="claude-3-5-haiku-latest")
    model_with_tools = model.bind_tools([get_weather, get_news])
    result = model_with_tools.invoke(messages)
    print(result.model_dump())
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatAnthropic.chat"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate AI message Event
    assert_message_in_logs(
        logs[1],
        "gen_ai.assistant.message",
        {
            "tool_calls": [
                {
                    "id": messages[1].tool_calls[0]["id"],
                    "function": {"name": messages[1].tool_calls[0]["name"]},
                    "type": "function",
                }
            ],
        },
    )

    # Validate tool message Event
    assert_message_in_logs(logs[2], "gen_ai.tool.message", {})

    # Validate AI choice Event
    result_dict = result.model_dump()
    choice_event = {
        "index": 0,
        "message": {},
        "finish_reason": "unknown",
        "tool_calls": [
            {
                "id": result_dict["content"][1]["id"],
                "function": {"name": result_dict["content"][1]["name"]},
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.vcr
def test_parallel_tool_calls(instrument_legacy, span_exporter, log_exporter):
    def get_weather(location: str) -> str:
        return "sunny"

    def get_news(location: str) -> str:
        return "Not much"

    messages: list[BaseMessage] = [
        HumanMessage(
            content="Hey, what's the weather in San Francisco? Also, any news in town?"
        ),
    ]
    model = ChatOpenAI(model="gpt-4.1-nano")
    model_with_tools = model.bind_tools([get_weather, get_news])
    model_with_tools.invoke(messages)
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatOpenAI.chat"

    assert chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name"] == "get_weather"
    assert json.loads(chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.parameters"]) == {
        "properties": {
            "location": {"type": "string"},
        },
        "required": ["location"],
        "type": "object",
    }

    assert chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.name"] == "get_news"
    assert json.loads(chat_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.1.parameters"]) == {
        "properties": {
            "location": {"type": "string"},
        },
        "required": ["location"],
        "type": "object",
    }

    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == messages[0].content
    )
    assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "user"

    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role"] == "assistant"
    )

    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.id"]
        == "call_EgULHWKqGjuB36aUeiOSpALZ"
    )
    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.name"]
        == "get_weather"
    )
    assert json.loads(
        chat_span.attributes[
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.0.arguments"
        ]
    ) == {"location": "San Francisco"}
    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.1.id"]
        == "call_Xer9QGOTDMG2Bxn9AKGiVM14"
    )
    assert (
        chat_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.1.name"]
        == "get_news"
    )
    assert json.loads(
        chat_span.attributes[
            f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.1.arguments"
        ]
    ) == {"location": "San Francisco"}

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.vcr
def test_parallel_tool_calls_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    def get_weather(location: str) -> str:
        return "sunny"

    def get_news(location: str) -> str:
        return "Not much"

    messages: list[BaseMessage] = [
        HumanMessage(
            content="Hey, what's the weather in San Francisco? Also, any news in town?"
        ),
    ]
    model = ChatOpenAI(model="gpt-4.1-nano")
    model_with_tools = model.bind_tools([get_weather, get_news])
    result = model_with_tools.invoke(messages)
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatOpenAI.chat"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(
        logs[0], "gen_ai.user.message", {"content": messages[0].content}
    )

    # Validate AI choice Event
    result_dict = result.model_dump()
    tool_calls = result_dict["additional_kwargs"]["tool_calls"]
    choice_event = {
        "index": 0,
        "message": {"content": result_dict["content"]},
        "finish_reason": "tool_calls",
        "tool_calls": [
            {
                "id": tool_calls[0]["id"],
                "function": {
                    "name": tool_calls[0]["function"]["name"],
                    "arguments": json.loads(tool_calls[0]["function"]["arguments"]),
                },
                "type": "function",
            },
            {
                "id": tool_calls[1]["id"],
                "function": {
                    "name": tool_calls[1]["function"]["name"],
                    "arguments": json.loads(tool_calls[1]["function"]["arguments"]),
                },
                "type": "function",
            },
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.skip(reason="Direct model invocations do not create langchain spans")
@pytest.mark.vcr
def test_parallel_tool_calls_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    def get_weather(location: str) -> str:
        return "sunny"

    def get_news(location: str) -> str:
        return "Not much"

    messages: list[BaseMessage] = [
        HumanMessage(
            content="Hey, what's the weather in San Francisco? Also, any news in town?"
        ),
    ]
    model = ChatOpenAI(model="gpt-4.1-nano")
    model_with_tools = model.bind_tools([get_weather, get_news])
    result = model_with_tools.invoke(messages)
    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    chat_span = spans[0]
    assert chat_span.name == "ChatOpenAI.chat"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate AI choice Event
    result_dict = result.model_dump()
    tool_calls = result_dict["additional_kwargs"]["tool_calls"]
    choice_event = {
        "index": 0,
        "message": {},
        "finish_reason": "tool_calls",
        "tool_calls": [
            {
                "id": tool_calls[0]["id"],
                "function": {"name": tool_calls[0]["function"]["name"]},
                "type": "function",
            },
            {
                "id": tool_calls[1]["id"],
                "function": {"name": tool_calls[1]["function"]["name"]},
                "type": "function",
            },
        ],
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "langchain"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
