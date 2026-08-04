"""Conformance of emitted spans against the OTel GenAI semantic conventions.

Warn-only until this package is tagged semconv:enforcing in project.json.

Reuses the existing cassette from `test_invocation.py` instead of recording a
new one: this test module needs no API keys to run.

There is no dedicated SageMaker contract group in the registry, so this checks
against the closest available AWS group, `aws.bedrock.inference.client`. That
group requires `aws.bedrock.guardrail.id`, which is Bedrock-specific and never
applies here — expect a `missing_required` warning for it on every span.

This package's instrumentation (span_utils.py) is minimal: it never sets
`gen_ai.operation.name`, `gen_ai.provider.name`, or `gen_ai.system` on the
span at all (the SDK's `event_emitter.py` only sets `gen_ai.system="sagemaker"`
on log *events*, not the span). `gen_ai.request.model` is the only span
attribute the shared harness can key on, so these tests pass
`identified_by="gen_ai.request.model"` instead of the default.
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
# not notice them going missing — which is exactly how #4362 (streaming drops
# finish_reasons) escaped detection. Adding a name here is a commitment.
#
# Note: `gen_ai.response.finish_reasons` is NOT emitted by this package at all —
# not even on log events, where `event_emitter.py` hardcodes `finish_reason`
# to the literal string "unknown" rather than reading it from the response.
# This is a full #4362-class gap, not just a streaming-path one.
EXPECTED = frozenset({"gen_ai.request.model"})


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    """Reuse the existing cassette rather than recording a new one."""
    return os.path.join(request.node.fspath.dirname, "cassettes")


@pytest.mark.vcr()
@pytest.mark.default_cassette(
    "test_invocation/test_sagemaker_completion_string_content_legacy"
)
def test_invocation_span_conforms(instrument_legacy, span_exporter, smrt):
    endpoint_name = "my-llama2-endpoint"
    prompt = "There's a llama in my garden. What should I do?"
    body = json.dumps(
        {
            "inputs": prompt,
            "parameters": {"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 128},
        }
    )
    smrt.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=body,
        ContentType="application/json",
    )

    check_exported_spans(
        span_exporter,
        CONTRACT_GROUP,
        enforcing=ENFORCING,
        expected=EXPECTED,
        identified_by="gen_ai.request.model",
    )
