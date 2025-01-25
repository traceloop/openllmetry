import pytest

from typing import List
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from opentelemetry.semconv_ai import SpanAttributes
from pydantic import BaseModel, Field


class FoodAnalysis(BaseModel):
    name: str = Field(description="The name of the food item")
    healthy: bool = Field(description="Whether the food is good for you")
    calories: int = Field(description="Estimated calories per serving")
    taste_profile: List[str] = Field(description="List of taste characteristics")


@pytest.mark.vcr
def test_structured_output(exporter):
    query_text = "Analyze the following food item: avocado"
    query = [HumanMessage(content=query_text)]
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    model_with_structured_output = model.with_structured_output(FoodAnalysis)
    result = model_with_structured_output.invoke(query)
    spans = exporter.get_finished_spans()

    span_names = set(span.name for span in spans)
    expected_spans = {"ChatOpenAI.chat"}
    assert expected_spans.issubset(span_names)

    chat_span = next(
        span for span in spans if span.name == "ChatOpenAI.chat"
    )

    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == query_text
    assert chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"] == result.model_dump_json()
