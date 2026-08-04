"""Conformance of emitted spans against the OTel GenAI semantic conventions.

Warn-only until this package is tagged semconv:enforcing in project.json.

Reuses an existing cassette from `test_completion.py` instead of recording a
new one: this test module needs no API keys to run.

This package never sets `gen_ai.operation.name` or `gen_ai.provider.name` on
its spans (see `__init__.py`); it only sets the legacy `gen_ai.system`
attribute. The shared harness keys spans on `gen_ai.operation.name` by
default, so this test passes `identified_by="gen_ai.system"` instead.
"""

import os

import pytest
from aleph_alpha_client import CompletionRequest, Prompt
from opentelemetry.semconv_ai._testing_conformance import check_exported_spans

# Keep in sync with the semconv:* tag in project.json.
ENFORCING = False

CONTRACT_GROUP = "span.gen_ai.inference.client"

# Attributes this package promises to emit on every inference span. No
# response model/id and no `gen_ai.response.finish_reasons` are ever set as
# span attributes (finish_reason only reaches the `gen_ai.choice` log event
# via event_emitter.py) -- a #4362-class gap.
EXPECTED = frozenset(
    {
        "gen_ai.request.model",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
    }
)


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    """Reuse existing cassettes rather than recording new ones."""
    return os.path.join(request.node.fspath.dirname, "cassettes")


@pytest.mark.vcr
@pytest.mark.default_cassette("test_completion/test_alephalpha_completion")
def test_completion_span_conforms(
    instrument_legacy, aleph_alpha_client, span_exporter
):
    prompt_text = "Tell me a joke about OpenTelemetry."
    request = CompletionRequest(
        prompt=Prompt.from_text(prompt_text), maximum_tokens=1000
    )
    aleph_alpha_client.complete(request, model="luminous-base")

    check_exported_spans(
        span_exporter,
        CONTRACT_GROUP,
        enforcing=ENFORCING,
        expected=EXPECTED,
        identified_by="gen_ai.system",
    )
