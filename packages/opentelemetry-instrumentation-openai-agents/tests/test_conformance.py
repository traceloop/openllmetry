"""Conformance of emitted spans against the OTel GenAI semantic conventions.

Warn-only until this package is tagged semconv:enforcing in project.json.

Reuses existing cassettes from `test_openai_agents.py` instead of recording
new ones: this test module needs no API keys to run.

This package is a framework instrumentation: a single `Runner.run_sync` call
emits several span kinds (the real inference span "openai.response", an
"<agent>.agent" span, an optional "<tool>.tool" span, and an "Agent Workflow"
root). Only the agent and response spans carry `gen_ai.operation.name` — the
workflow span never sets it — so the shared harness (keyed on that attribute
by default) checks both of those against CONTRACT_GROUP here. The agent span
is missing most of the inference-only attributes in EXPECTED, which surfaces
as extra warnings: expected noise, not a bug in this test.

`gen_ai.request.model` is notably absent from the response span itself (only
`gen_ai.response.model` is set), so it is deliberately excluded from EXPECTED.
"""

import os

import pytest
from agents import Runner
from opentelemetry.semconv_ai._testing_conformance import check_exported_spans

# Keep in sync with the semconv:* tag in project.json.
ENFORCING = False

CONTRACT_GROUP = "openai.inference.client"

# Attributes this package promises to emit on every inference span. The registry
# marks most of these merely `recommended`, so without this set the harness would
# not notice them going missing — which is exactly how #4362 (streaming drops
# finish_reasons) escaped detection. Adding a name here is a commitment.
EXPECTED = frozenset(
    {
        "gen_ai.operation.name",
        "gen_ai.provider.name",
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
@pytest.mark.default_cassette("test_openai_agents/test_agent_spans")
def test_agent_response_span_conforms(exporter, test_agent):
    query = "What is AI?"
    Runner.run_sync(test_agent, query)

    check_exported_spans(
        exporter, CONTRACT_GROUP, enforcing=ENFORCING, expected=EXPECTED
    )


@pytest.mark.vcr
@pytest.mark.default_cassette("test_openai_agents/test_agent_with_function_tool_spans")
def test_agent_with_tool_call_response_spans_conform(exporter, function_tool_agent):
    """Tool-calling produces two response spans (before and after the tool call)."""
    query = "What is the weather in London?"
    Runner.run_sync(function_tool_agent, query)

    check_exported_spans(
        exporter, CONTRACT_GROUP, enforcing=ENFORCING, expected=EXPECTED
    )
