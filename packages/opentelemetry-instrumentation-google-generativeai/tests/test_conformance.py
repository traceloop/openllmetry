"""Conformance of emitted spans against the OTel GenAI semantic conventions.

Warn-only until this package is tagged semconv:enforcing in project.json.

Reuses an existing cassette from `test_generate_content.py` instead of
recording a new one: this test module needs no API keys to run.

Unlike the other five packages in this rollout, this instrumentation already
targets the newer GenAI semantic conventions directly: it sets
`gen_ai.operation.name` and `gen_ai.provider.name` on every span (see
`span_utils.py`), so the default `identified_by="gen_ai.operation.name"` finds
its spans without an override.
"""

import os

import pytest
from opentelemetry.semconv_ai._testing_conformance import check_exported_spans

# Keep in sync with the semconv:* tag in project.json.
ENFORCING = False

CONTRACT_GROUP = "span.gen_ai.inference.client"

# Attributes this package promises to emit on every inference span. This is
# the most modern of the six packages in this rollout: it already emits
# gen_ai.response.finish_reasons, gen_ai.response.id, gen_ai.input.messages,
# and gen_ai.output.messages -- attributes several of the other five never
# set at all.
EXPECTED = frozenset(
    {
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.response.id",
        "gen_ai.response.finish_reasons",
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.total_tokens",
    }
)


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    """Reuse existing cassettes rather than recording new ones."""
    return os.path.join(request.node.fspath.dirname, "cassettes")


@pytest.mark.vcr
@pytest.mark.default_cassette("test_generate_content/test_client_spans")
def test_generate_content_span_conforms(exporter, genai_client):
    genai_client.chats.create(model="gemini-2.5-flash").send_message("What is ai?")

    check_exported_spans(
        exporter,
        CONTRACT_GROUP,
        enforcing=ENFORCING,
        expected=EXPECTED,
    )
