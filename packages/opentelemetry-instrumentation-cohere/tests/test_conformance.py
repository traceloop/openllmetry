"""Conformance of emitted spans against the OTel GenAI semantic conventions.

Warn-only until this package is tagged semconv:enforcing in project.json.

Reuses existing cassettes from `test_chat.py` instead of recording new ones:
this test module needs no API keys to run.

This package never sets `gen_ai.operation.name` or `gen_ai.provider.name` on its
spans (see `span_utils.py` / `__init__.py`); it only sets the legacy
`gen_ai.system` attribute. The shared harness keys spans on
`gen_ai.operation.name` by default, so these tests pass
`identified_by="gen_ai.system"` instead.
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
# Unlike ollama, this package DOES emit `gen_ai.response.finish_reasons`
# (span_utils.py:_set_span_chat_response). Token usage, however, is only
# reported under the legacy `gen_ai.usage.{prompt,completion}_tokens` names,
# which are neither contract attributes nor declared extensions, so they are
# excluded here.
EXPECTED = frozenset(
    {
        "gen_ai.request.model",
        "gen_ai.response.id",
        "gen_ai.response.finish_reasons",
    }
)


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    """Reuse existing cassettes rather than recording new ones."""
    return os.path.join(request.node.fspath.dirname, "cassettes")


@pytest.mark.vcr
@pytest.mark.default_cassette("test_chat/test_cohere_chat_legacy")
def test_chat_span_conforms(instrument_legacy, cohere_client, span_exporter):
    cohere_client.chat(model="command", message="Tell me a joke, pirate style")
    check_exported_spans(
        span_exporter,
        CONTRACT_GROUP,
        enforcing=ENFORCING,
        expected=EXPECTED,
        identified_by="gen_ai.system",
    )


@pytest.mark.vcr
@pytest.mark.default_cassette("test_chat/test_cohere_v2_chat_legacy")
def test_v2_chat_span_conforms(instrument_legacy, cohere_client_v2, span_exporter):
    cohere_client_v2.chat(
        model="command",
        messages=[{"role": "user", "content": "Tell me a joke, pirate style"}],
    )
    check_exported_spans(
        span_exporter,
        CONTRACT_GROUP,
        enforcing=ENFORCING,
        expected=EXPECTED,
        identified_by="gen_ai.system",
    )
