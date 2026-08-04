"""Conformance of emitted spans against the OTel GenAI semantic conventions.

Warn-only until this package is tagged semconv:enforcing in project.json.

Reuses existing cassettes from `test_structured_llm.py` instead of recording
new ones: this test module needs no API keys to run.

This package is a framework instrumentation: a single `StructuredLLM.chat`
call emits both the real inference span (`openai.chat`, delegated to the
OpenAI instrumentation) and a `StructuredLLM.workflow` span. Both carry
`gen_ai.operation.name`, so the shared harness (keyed on that attribute by
default) checks both against CONTRACT_GROUP here. The workflow span is
missing most of the inference-only attributes in EXPECTED, which surfaces as
extra warnings — this is expected noise, not a bug in this test.
"""

import os

import pytest
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from opentelemetry.semconv_ai._testing_conformance import check_exported_spans
from pydantic import BaseModel, Field

# Keep in sync with the semconv:* tag in project.json.
ENFORCING = False

CONTRACT_GROUP = "span.gen_ai.inference.client"

# Attributes this package promises to emit on every inference span. The registry
# marks most of these merely `recommended`, so without this set the harness would
# not notice them going missing — which is exactly how #4362 (streaming drops
# finish_reasons) escaped detection. Adding a name here is a commitment.
EXPECTED = frozenset(
    {
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.response.finish_reasons",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
    }
)


class Invoice(BaseModel):
    """Example model for structured output testing."""

    invoice_id: str = Field(description="Invoice identifier")
    amount: float = Field(description="Invoice amount")
    customer_name: str = Field(description="Customer name")


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    """Reuse existing cassettes rather than recording new ones."""
    return os.path.join(request.node.fspath.dirname, "cassettes")


@pytest.mark.vcr()
@pytest.mark.default_cassette(
    "test_structured_llm/test_structured_llm_model_attributes"
)
def test_structured_llm_chat_span_conforms(instrument_with_content, span_exporter):
    llm = OpenAI(model="gpt-4o", temperature=0.7)
    structured_llm = llm.as_structured_llm(Invoice)

    messages = [
        ChatMessage(
            role="system",
            content="Extract invoice information from the following text.",
        ),
        ChatMessage(role="user", content="Invoice #12345 for $199.99 to John Smith"),
    ]
    structured_llm.chat(messages)

    check_exported_spans(
        span_exporter, CONTRACT_GROUP, enforcing=ENFORCING, expected=EXPECTED
    )


@pytest.mark.vcr()
@pytest.mark.default_cassette(
    "test_structured_llm/test_structured_llm_achat_model_attributes"
)
@pytest.mark.asyncio
async def test_structured_llm_achat_span_conforms(instrument_with_content, span_exporter):
    """Async path; #4362 is a streaming/async-code-path class of gap."""
    llm = OpenAI(model="gpt-4o", temperature=0.5)
    structured_llm = llm.as_structured_llm(Invoice)

    messages = [
        ChatMessage(
            role="system",
            content="Extract invoice information from the following text.",
        ),
        ChatMessage(role="user", content="Invoice #67890 for $299.99 to Jane Doe"),
    ]
    await structured_llm.achat(messages)

    check_exported_spans(
        span_exporter, CONTRACT_GROUP, enforcing=ENFORCING, expected=EXPECTED
    )
