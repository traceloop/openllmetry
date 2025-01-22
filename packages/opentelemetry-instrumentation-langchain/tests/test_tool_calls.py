import pytest

from typing import List
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from opentelemetry.semconv_ai import SpanAttributes


def food_analysis(name: str, healthy: bool, calories: int, taste_profile: List[str]) -> str:
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
