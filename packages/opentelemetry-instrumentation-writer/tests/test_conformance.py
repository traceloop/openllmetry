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
# Note: `gen_ai.response.finish_reasons` is NOT emitted by this package in either
# the legacy or streaming path (span_utils.py only sets the legacy nested
# `gen_ai.completion.0.finish_reason`), so it is deliberately absent from
# EXPECTED — a #4362-class gap.
EXPECTED = frozenset(
    {
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.request.max_tokens",
        "gen_ai.request.temperature",
        "gen_ai.request.top_p",
    }
)


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    """Reuse existing cassettes rather than recording new ones."""
    return os.path.join(request.node.fspath.dirname, "cassettes")


@pytest.mark.vcr
@pytest.mark.default_cassette("test_chat/test_writer_chat_legacy")
def test_chat_span_conforms(instrument_legacy, writer_client, span_exporter):
    writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
    )
    check_exported_spans(
        span_exporter,
        CONTRACT_GROUP,
        enforcing=ENFORCING,
        expected=EXPECTED,
        identified_by="gen_ai.system",
    )


@pytest.mark.vcr
@pytest.mark.default_cassette("test_chat/test_writer_streaming_chat_legacy")
def test_streaming_span_conforms(instrument_legacy, writer_client, span_exporter):
    """Streaming is where #4362 reports dropped attributes."""
    gen = writer_client.chat.chat(
        model="palmyra-x4",
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=340,
        stop="I am",
        stream=True,
        stream_options={"include_usage": True},
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
