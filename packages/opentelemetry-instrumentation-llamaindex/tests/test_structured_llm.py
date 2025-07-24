import pytest
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from opentelemetry.semconv_ai import SpanAttributes, LLMRequestTypeValues


class Invoice(BaseModel):
    """Example model for structured output testing."""

    invoice_id: str = Field(description="Invoice identifier")
    amount: float = Field(description="Invoice amount")
    customer_name: str = Field(description="Customer name")


@pytest.mark.vcr()
def test_structured_llm_model_attributes(instrument_with_content, span_exporter):
    """
    Test that StructuredLLM correctly sets model attributes.

    This test reproduces the issue where set_llm_chat_request_model_attributes
    fails to access model and temperature from StructuredLLM because it tries
    to access model_dict.model instead of model_dict.llm.model.
    """
    llm = OpenAI(model="gpt-4o", temperature=0.7)
    structured_llm = llm.as_structured_llm(Invoice)

    messages = [
        ChatMessage(
            role="system",
            content="Extract invoice information from the following text.",
        ),
        ChatMessage(role="user", content="Invoice #12345 for $199.99 to John Smith"),
    ]

    response = structured_llm.chat(messages)

    assert response is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) > 0

    llm_span = None
    for span in spans:
        if (
            span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE)
            == LLMRequestTypeValues.CHAT.value
        ):
            llm_span = span
            break

    assert llm_span is not None, "Should have an LLM span"

    attributes = llm_span.attributes
    assert "gen_ai.request.model" in attributes
    assert attributes["gen_ai.request.model"] == "gpt-4o"
    assert "gen_ai.request.temperature" in attributes
    assert attributes["gen_ai.request.temperature"] == 0.7


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_structured_llm_achat_model_attributes(
    instrument_with_content, span_exporter
):
    """
    Test that StructuredLLM achat method correctly sets model attributes.

    This is the async version of the test that reproduces the original issue.
    """
    llm = OpenAI(model="gpt-4o", temperature=0.5)
    structured_llm = llm.as_structured_llm(Invoice)

    messages = [
        ChatMessage(
            role="system",
            content="Extract invoice information from the following text.",
        ),
        ChatMessage(role="user", content="Invoice #67890 for $299.99 to Jane Doe"),
    ]

    response = await structured_llm.achat(messages)

    assert response is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) > 0

    llm_span = None
    for span in spans:
        if (
            span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE)
            == LLMRequestTypeValues.CHAT.value
        ):
            llm_span = span
            break

    assert llm_span is not None, "Should have an LLM span"

    attributes = llm_span.attributes
    assert "gen_ai.request.model" in attributes
    assert attributes["gen_ai.request.model"] == "gpt-4o"
    assert "gen_ai.request.temperature" in attributes
    assert attributes["gen_ai.request.temperature"] == 0.5
