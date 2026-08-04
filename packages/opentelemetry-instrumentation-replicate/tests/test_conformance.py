"""Conformance of emitted spans against the OTel GenAI semantic conventions.

Warn-only until this package is tagged semconv:enforcing in project.json.

Reuses an existing cassette from `test_llama.py` instead of recording a new
one: this test module needs no API keys to run.

This package never sets `gen_ai.operation.name` or `gen_ai.provider.name` on
its spans (see `__init__.py`); it only sets the legacy `gen_ai.system`
attribute at span creation. The shared harness keys spans on
`gen_ai.operation.name` by default, so this test passes
`identified_by="gen_ai.system"` instead.
"""

import os

import pytest
from opentelemetry.semconv_ai._testing_conformance import check_exported_spans

# Keep in sync with the semconv:* tag in project.json.
ENFORCING = False

CONTRACT_GROUP = "span.gen_ai.inference.client"

# Attributes this package promises to emit on every inference span. Of the
# whole contract, this package only ever sets `gen_ai.request.model`
# (span_utils.py:set_model_input_attributes) — no response model/id, no token
# usage, and no `gen_ai.response.finish_reasons` are ever recorded as span
# attributes. This is the sparsest of the six packages in this rollout.
EXPECTED = frozenset(
    {
        "gen_ai.request.model",
    }
)


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    """Reuse existing cassettes rather than recording new ones."""
    return os.path.join(request.node.fspath.dirname, "cassettes")


@pytest.mark.vcr
@pytest.mark.default_cassette("test_llama/test_replicate_llama_stream_legacy")
def test_stream_span_conforms(instrument_legacy, replicate_client, span_exporter):
    model_version = "meta/llama-2-70b-chat"
    for event in replicate_client.stream(
        model_version,
        input={"prompt": "tell me a joke about opentelemetry"},
    ):
        continue

    check_exported_spans(
        span_exporter,
        CONTRACT_GROUP,
        enforcing=ENFORCING,
        expected=EXPECTED,
        identified_by="gen_ai.system",
    )
