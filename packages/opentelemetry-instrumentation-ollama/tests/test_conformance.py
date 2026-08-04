"""Conformance of emitted spans against the OTel GenAI semantic conventions.

Warn-only until this package is tagged semconv:enforcing in project.json.

Reuses existing cassettes from `test_chat.py` instead of recording new ones:
this test module needs no API keys to run.

This package never sets `gen_ai.operation.name` or `gen_ai.provider.name` on its
spans (see `span_utils.py`); it only sets the legacy `gen_ai.system` attribute.
The shared harness keys spans on `gen_ai.operation.name` by default, so these
tests pass `identified_by="gen_ai.system"` instead.
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
# `gen_ai.response.finish_reasons` is set from Ollama's `done_reason` in
# span_utils.py (set_model_response_attributes), in both the legacy and
# streaming paths — closing the #4362-class gap.
EXPECTED = frozenset(
    {
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
@pytest.mark.default_cassette("test_chat/test_ollama_chat_legacy")
def test_chat_span_conforms(instrument_legacy, ollama_client, span_exporter):
    ollama_client.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
    )
    check_exported_spans(
        span_exporter,
        CONTRACT_GROUP,
        enforcing=ENFORCING,
        expected=EXPECTED,
        identified_by="gen_ai.system",
    )


@pytest.mark.vcr
@pytest.mark.default_cassette("test_chat/test_ollama_streaming_chat_legacy")
def test_streaming_span_conforms(instrument_legacy, ollama_client, span_exporter):
    """Streaming is where #4362 reports dropped attributes."""
    gen = ollama_client.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        stream=True,
    )
    for _ in gen:
        pass

    check_exported_spans(
        span_exporter,
        CONTRACT_GROUP,
        enforcing=ENFORCING,
        expected=EXPECTED,
        identified_by="gen_ai.system",
    )
