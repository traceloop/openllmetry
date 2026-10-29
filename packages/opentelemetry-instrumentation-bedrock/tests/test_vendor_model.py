"""Unit tests for inference profile id → (vendor, model) resolution.

These are pure unit tests (no VCR cassettes, no AWS clients) that validate
`_get_vendor_model` in `__init__.py` against the id shapes Bedrock's system
inference profiles use.
"""

import pytest

from opentelemetry.instrumentation.bedrock import _get_vendor_model

AWS_BEDROCK = "aws.bedrock"


@pytest.mark.parametrize(
    "model_id, expected",
    [
        # Geo-prefixed ids whose model name itself contains a dot.
        ("us.openai.gpt-5.6-sol", (AWS_BEDROCK, "openai", "gpt-5.6-sol")),
        ("us.openai.gpt-5.6-luna", (AWS_BEDROCK, "openai", "gpt-5.6-luna")),
        ("us.xai.grok-4.6", (AWS_BEDROCK, "xai", "grok-4.6")),
        ("eu.anthropic.claude-sonnet-4-5-20250929-v1:0", (AWS_BEDROCK, "anthropic", "claude-sonnet-4-5-20250929-v1:0")),
        ("apac.amazon.nova-pro-v1:0", (AWS_BEDROCK, "amazon", "nova-pro-v1:0")),
        ("us-gov.anthropic.claude-3-5-sonnet-20241022-v2:0",
         (AWS_BEDROCK, "anthropic", "claude-3-5-sonnet-20241022-v2:0")),
        # Geo-prefixed ids with no dot in the model name.
        ("us.openai.gpt-6-astra", (AWS_BEDROCK, "openai", "gpt-6-astra")),
        ("us.anthropic.claude-3-5-haiku-20241022-v1:0", (AWS_BEDROCK, "anthropic", "claude-3-5-haiku-20241022-v1:0")),
        # Same id arriving as an inference profile ARN.
        (
            "arn:aws:bedrock:us-east-1:111122223333:inference-profile/us.openai.gpt-5.6-sol",
            (AWS_BEDROCK, "openai", "gpt-5.6-sol"),
        ),
        # Unprefixed ids are resolved as-is.
        ("openai.gpt-5.6-sol", (AWS_BEDROCK, "openai", "gpt-5.6-sol")),
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", (AWS_BEDROCK, "anthropic", "claude-3-5-sonnet-20241022-v2:0")),
        # On-demand model ids with no vendor prefix are reported verbatim.
        ("nova-pro-v1:0", (AWS_BEDROCK, "imported_model", "nova-pro-v1:0")),
        (None, (AWS_BEDROCK, "imported_model", None)),
    ],
)
def test_get_vendor_model(model_id, expected):
    assert _get_vendor_model(model_id) == expected
