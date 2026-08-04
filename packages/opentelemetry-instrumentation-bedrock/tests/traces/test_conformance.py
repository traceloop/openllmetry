"""Conformance of emitted spans against the OTel GenAI semantic conventions.

Warn-only until this package is tagged semconv:enforcing in project.json.

Reuses existing cassettes from `test_anthropic.py` instead of recording new
ones: this test module needs no AWS credentials to run.

Lives under `tests/traces/` (not `tests/test_conformance.py`) so its default
cassette directory (`tests/traces/cassettes`) matches where this package's
trace cassettes actually live; fixtures (`brt`, `instrument_legacy`,
`vcr_config`) come from the parent `tests/conftest.py` — bedrock has no
`tests/traces/conftest.py` of its own.
"""

import json
import os

import pytest
from opentelemetry.semconv_ai._testing_conformance import check_exported_spans

# Keep in sync with the semconv:* tag in project.json.
ENFORCING = False

CONTRACT_GROUP = "aws.bedrock.inference.client"

# Attributes this package promises to emit on every inference span. The registry
# marks most of these merely `recommended`, so without this set the harness would
# not notice them going missing. Adding a name here is a commitment.
#
# `gen_ai.response.model` and `gen_ai.response.id` are deliberately excluded:
# the legacy Claude-2 completion API (test_anthropic_2_completion) never returns
# a response model/id, so they are not emitted on every span, only Claude-3+.
#
# Note: `gen_ai.response.finish_reasons` IS emitted here (both the legacy
# text_completion path and the Claude-3 streaming/chat path) — no #4362-class
# gap for this package.
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
@pytest.mark.default_cassette("test_anthropic/test_anthropic_2_completion")
def test_completion_span_conforms(instrument_legacy, brt, span_exporter):
    body = json.dumps(
        {
            "prompt": "Human: Tell me a joke about opentelemetry Assistant:",
            "max_tokens_to_sample": 200,
            "temperature": 0.5,
        }
    )
    brt.invoke_model(
        body=body,
        modelId="anthropic.claude-v2:1",
        accept="application/json",
        contentType="application/json",
    )
    check_exported_spans(
        span_exporter, CONTRACT_GROUP, enforcing=ENFORCING, expected=EXPECTED
    )


@pytest.mark.vcr
@pytest.mark.default_cassette("test_anthropic/test_anthropic_3_completion_streaming")
def test_streaming_span_conforms(instrument_legacy, brt, span_exporter):
    """Streaming is where #4362 reports dropped attributes."""
    body = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Tell me a joke about opentelemetry"},
                    ],
                }
            ],
            "max_tokens": 200,
            "temperature": 0.5,
            "anthropic_version": "bedrock-2023-05-31",
        }
    )
    response = brt.invoke_model_with_response_stream(
        body=body,
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        accept="application/json",
        contentType="application/json",
    )
    for _ in response.get("body"):
        pass

    check_exported_spans(
        span_exporter, CONTRACT_GROUP, enforcing=ENFORCING, expected=EXPECTED
    )
