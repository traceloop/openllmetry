"""Conformance of emitted spans against the OTel GenAI semantic conventions.

Warn-only until this package is tagged semconv:enforcing in project.json.

Reuses existing cassettes from `test_chat_tracing.py` instead of recording new
ones: this test module needs no API keys to run.

Lives under `tests/traces/` (matching where this package's only conftest.py
and cassettes already live; there is no separate `tests/conftest.py` here).
"""

import os

import pytest
from opentelemetry.semconv_ai._testing_conformance import check_exported_spans

# Keep in sync with the semconv:* tag in project.json.
ENFORCING = False

CONTRACT_GROUP = "span.gen_ai.inference.client"

# Attributes this package promises to emit on every inference span. The registry
# marks most of these merely `recommended`, so without this set the harness would
# not notice them going missing. Adding a name here is a commitment.
#
# `gen_ai.response.model` and `gen_ai.response.id` are deliberately excluded:
# they are present on the non-streaming span but absent from the streaming one.
#
# Note: `gen_ai.response.finish_reasons` IS emitted here (both legacy and
# streaming paths) — no #4362-class gap for this package.
EXPECTED = frozenset(
    {
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.request.model",
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
@pytest.mark.default_cassette("test_chat_tracing/test_chat_legacy")
def test_chat_span_conforms(instrument_legacy, groq_client, span_exporter):
    groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": "Tell me a joke about opentelemetry"},
        ],
    )
    check_exported_spans(
        span_exporter, CONTRACT_GROUP, enforcing=ENFORCING, expected=EXPECTED
    )


@pytest.mark.vcr
@pytest.mark.default_cassette("test_chat_tracing/test_chat_streaming_legacy")
def test_streaming_span_conforms(instrument_legacy, groq_client, span_exporter):
    """Streaming is where #4362 reports dropped attributes."""
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": "Tell me a joke about opentelemetry"},
        ],
        stream=True,
    )
    for _ in response:
        pass

    check_exported_spans(
        span_exporter, CONTRACT_GROUP, enforcing=ENFORCING, expected=EXPECTED
    )
