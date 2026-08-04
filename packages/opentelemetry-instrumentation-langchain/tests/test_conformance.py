"""Conformance of emitted spans against the OTel GenAI semantic conventions.

Warn-only until this package is tagged semconv:enforcing in project.json.

Reuses existing cassettes from `test_llms.py` instead of recording new ones:
this test module needs no API keys to run.

This package is a framework instrumentation: a single chain invocation emits
several span kinds ("execute_task ChatPromptTemplate", the model's own
inference span, and "RunnableSequence.workflow"), and all of them carry
`gen_ai.operation.name` (with different values: execute_task, chat,
invoke_agent). The shared harness keys spans on that attribute by default, so
every one of those spans gets checked against CONTRACT_GROUP here, not just
the inference span. The non-inference spans are missing most of the
inference-only attributes in EXPECTED, which surfaces as extra warnings — this
is expected noise, not a bug in this test.
"""

import os

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from opentelemetry.semconv_ai._testing_conformance import check_exported_spans

from tests.test_llms import open_ai_prompt

# Keep in sync with the semconv:* tag in project.json.
ENFORCING = False

CONTRACT_GROUP = "span.gen_ai.inference.client"

# Attributes this package promises to emit on every inference span. The registry
# marks most of these merely `recommended`, so without this set the harness would
# not notice them going missing — which is exactly how #4362 (streaming drops
# finish_reasons) escaped detection. Adding a name here is a commitment.
#
# `gen_ai.response.finish_reasons` is provider-dependent here: the OpenAI-backed
# span (test_openai) carries it, but the Anthropic-backed span (test_anthropic)
# does not — langchain only ever nests a `finish_reason` inside
# `gen_ai.output.messages` for Anthropic, never promoting it to the top-level
# attribute. It stays in EXPECTED to lock in and surface the OpenAI-path
# behavior; the Anthropic test's resulting `missing_expected` warning is a real,
# #4362-class gap, not a mistake in this test.
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


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    """Reuse existing cassettes rather than recording new ones."""
    return os.path.join(request.node.fspath.dirname, "cassettes")


@pytest.mark.vcr
@pytest.mark.default_cassette("test_llms/test_openai")
def test_openai_chat_span_conforms(instrument_legacy, span_exporter, log_exporter):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("human", "{input}")]
    )
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model
    chain.invoke({"input": open_ai_prompt()})

    check_exported_spans(
        span_exporter, CONTRACT_GROUP, enforcing=ENFORCING, expected=EXPECTED
    )


@pytest.mark.vcr
@pytest.mark.default_cassette("test_llms/test_anthropic")
def test_anthropic_chat_span_conforms(instrument_legacy, span_exporter, log_exporter):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("user", "{input}")]
    )
    model = ChatAnthropic(model="claude-2.1", temperature=0.5)
    chain = prompt | model
    chain.invoke({"input": "tell me a short joke"})

    check_exported_spans(
        span_exporter, CONTRACT_GROUP, enforcing=ENFORCING, expected=EXPECTED
    )
