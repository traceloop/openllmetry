from typing import List

import pytest
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    event_attributes as EventAttributes,
)
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
    result = model_with_tools.invoke(query)
    spans = span_exporter.get_finished_spans()

    span_names = set(span.name for span in spans)
    expected_spans = {"ChatOpenAI.chat"}
    assert expected_spans.issubset(span_names)

    chat_span = next(span for span in spans if span.name == "ChatOpenAI.chat")

    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == query_text
    assert (
        chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name"]
        == "food_analysis"
    )

    arguments = chat_span.attributes[
        f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments"
    ]
    assert arguments == result.model_dump().get("additional_kwargs").get("tool_calls")[
        0
    ].get("function").get("arguments")

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


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


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.attributes.get(EventAttributes.EVENT_NAME) == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "langchain"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
