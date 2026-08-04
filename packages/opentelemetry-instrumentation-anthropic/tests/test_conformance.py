"""Conformance of emitted spans against the OTel GenAI semantic conventions.

Warn-only until this package is tagged semconv:enforcing in project.json.

Reuses existing cassettes from `test_messages.py` and `test_structured_outputs.py`
instead of recording new ones: this test module needs no API keys to run.

WARNING: cassette matching here is unguarded. VCR's default `match_on` compares
only (method, scheme, host, port, path, query) — the request body is not checked,
and every call in this package posts to the same `/v1/messages` path. Matching on
`body` was tried to close this gap but proved brittle: the installed anthropic SDK
rewrites `output_format` into `output_config.format` on the wire independent of what
was recorded, so a byte-for-byte body match breaks across SDK versions even when
nothing about the test's intent changed. If you change a source test referenced by
a `default_cassette` marker below (model, prompt, or request shape), you must mirror
that change here too — otherwise this module will keep silently replaying the old,
stale cassette instead of failing.
"""

import os

import pytest
from opentelemetry.semconv_ai._testing_conformance import check_exported_spans

# Keep in sync with the semconv:* tag in project.json.
ENFORCING = False

CONTRACT_GROUP = "anthropic.inference.client"

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


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    """Reuse existing cassettes rather than recording new ones.

    Rooted at `tests/cassettes` (not a single test module's subdirectory) so
    `default_cassette` markers below can reach into whichever sibling test
    module's cassette they need, e.g. "test_structured_outputs/<name>".
    """
    return os.path.join(request.node.fspath.dirname, "cassettes")


@pytest.mark.vcr
@pytest.mark.default_cassette("test_messages/test_anthropic_message_create_legacy")
def test_messages_span_conforms(instrument_legacy, anthropic_client, span_exporter):
    anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-opus-20240229",
    )
    check_exported_spans(
        span_exporter, CONTRACT_GROUP, enforcing=ENFORCING, expected=EXPECTED
    )


@pytest.mark.vcr
@pytest.mark.default_cassette("test_messages/test_anthropic_message_streaming_legacy")
def test_streaming_span_conforms(instrument_legacy, anthropic_client, span_exporter):
    """Streaming is where #4362 reports dropped attributes."""
    stream = anthropic_client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="claude-3-haiku-20240307",
        stream=True,
    )
    for event in stream:
        pass

    check_exported_spans(
        span_exporter, CONTRACT_GROUP, enforcing=ENFORCING, expected=EXPECTED
    )


@pytest.mark.vcr
@pytest.mark.default_cassette(
    "test_structured_outputs/test_anthropic_structured_outputs_legacy"
)
def test_structured_output_span_conforms(
    instrument_legacy, anthropic_client, span_exporter
):
    """Structured-output requests emit `gen_ai.request.structured_output_schema`,
    which the contract does not define and `extensions.py` does not declare.

    This is a real, undeclared `gen_ai.*` attribute (see
    `opentelemetry/instrumentation/anthropic/span_utils.py`), not a fixture or
    contract-group mistake, and is the proof that this harness actually detects
    violations rather than passing vacuously (see Step 4 verification).
    """
    output_format = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "joke": {
                    "type": "string",
                    "description": "A joke about OpenTelemetry",
                },
                "rating": {
                    "type": "integer",
                    "description": "Rating of the joke from 1 to 10",
                },
            },
            "required": ["joke", "rating"],
            "additionalProperties": False,
        },
    }
    anthropic_client.beta.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        betas=["structured-outputs-2025-11-13"],
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry and rate it from 1 to 10",
            }
        ],
        output_format=output_format,
    )

    check_exported_spans(
        span_exporter, CONTRACT_GROUP, enforcing=ENFORCING, expected=EXPECTED
    )
