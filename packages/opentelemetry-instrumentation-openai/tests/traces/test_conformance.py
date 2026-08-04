"""Conformance of emitted spans against the OTel GenAI semantic conventions.

Warn-only until this package is tagged semconv:enforcing in project.json.

Reuses existing cassettes from `test_chat.py` instead of recording new ones:
this test module needs no API keys to run.

Lives under `tests/traces/` (not `tests/test_conformance.py`) because this
package splits fixtures across `tests/conftest.py` (clients, instrumentors)
and `tests/traces/conftest.py` (the `vcr_config` used for trace cassettes) —
the trace-cassette fixtures are only in scope here.
"""

import os

import pytest
from opentelemetry.semconv_ai._testing_conformance import check_exported_spans

# Keep in sync with the semconv:* tag in project.json.
ENFORCING = False

CONTRACT_GROUP = "openai.inference.client"

# Attributes this package promises to emit on every inference span. The registry
# marks most of these merely `recommended`, so without this set the harness would
# not notice them going missing. Adding a name here is a commitment.
#
# Note: `gen_ai.response.finish_reasons` IS emitted here (both legacy and
# streaming paths), unlike several other packages in this rollout — no #4362-class
# gap for this package.
EXPECTED = frozenset(
    {
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.response.id",
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
@pytest.mark.default_cassette("test_chat/test_chat")
def test_chat_span_conforms(instrument_legacy, span_exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Tell me a joke about opentelemetry"},
        ],
    )
    check_exported_spans(
        span_exporter, CONTRACT_GROUP, enforcing=ENFORCING, expected=EXPECTED
    )


@pytest.mark.vcr
@pytest.mark.default_cassette("test_chat/test_chat_streaming")
def test_streaming_span_conforms(instrument_legacy, span_exporter, mock_openai_client):
    """Streaming is where #4362 reports dropped attributes."""
    response = mock_openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
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
