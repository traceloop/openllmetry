import pytest
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from opentelemetry.semconv_ai import SpanAttributes
import json

# Define the test schema using Pydantic
class FoodAnalysis(BaseModel):
    name: str = Field(description="The name of the food item")
    healthy: bool = Field(description="Whether the food is good for you")
    calories: int = Field(description="Estimated calories per serving")
    taste_profile: List[str] = Field(description="List of taste characteristics")

@pytest.mark.vcr
def test_structured_output(exporter):
    query = "Analyze the following food item: avocado"
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_model = model.with_structured_output(FoodAnalysis)
    result = structured_model.invoke(query)
    spans = exporter.get_finished_spans()
    
    span_names = set(span.name for span in spans)
    expected_spans = {"ChatOpenAI.chat"}
    assert expected_spans.issubset(span_names)

    chat_span = next(
        span for span in spans if span.name == "ChatOpenAI.chat"
    )
    
    assert chat_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"] == query
    assert chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name"] == "FoodAnalysis"
    assert chat_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments"] == result.model_dump_json()
