from typing import List

import pytest
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from pydantic import BaseModel, Field


class FoodAnalysis(BaseModel):
    name: str = Field(description="The name of the food item")
    healthy: bool = Field(description="Whether the food is good for you")
    calories: int = Field(description="Estimated calories per serving")
    taste_profile: List[str] = Field(description="List of taste characteristics")


@pytest.mark.vcr
def test_structured_output(instrument_legacy, span_exporter, log_exporter):
    query_text = "Analyze the following food item: avocado"
    query = [HumanMessage(content=query_text)]
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    model_with_structured_output = model.with_structured_output(FoodAnalysis)
    _ = model_with_structured_output.invoke(query)
    spans = span_exporter.get_finished_spans()

    span_names = set(span.name for span in spans)
    expected_spans = {"ChatOpenAI.chat"}
    assert expected_spans.issubset(span_names)

    chat_span = next(span for span in spans if span.name == "ChatOpenAI.chat")

    assert chat_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"] == query_text

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_structured_output_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    query_text = "Analyze the following food item: avocado"
    query = [HumanMessage(content=query_text)]
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    model_with_structured_output = model.with_structured_output(FoodAnalysis)
    _result = model_with_structured_output.invoke(query)
    spans = span_exporter.get_finished_spans()

    span_names = set(span.name for span in spans)
    expected_spans = {"ChatOpenAI.chat"}
    assert expected_spans.issubset(span_names)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {"content": query_text})

    assert _result != ""

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": _result.model_dump_json()},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_structured_output_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    query_text = "Analyze the following food item: avocado"
    query = [HumanMessage(content=query_text)]
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    model_with_structured_output = model.with_structured_output(FoodAnalysis)
    model_with_structured_output.invoke(query)
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
        "finish_reason": "stop",
        "message": {},
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
