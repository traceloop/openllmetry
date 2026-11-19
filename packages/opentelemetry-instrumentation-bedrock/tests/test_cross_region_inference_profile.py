"""Cross-region inference profile parsing tests.

Covers _cross_region_check / _get_vendor_model — the pure functions that derive
(provider, model_vendor, model) from a Bedrock modelId. The model_vendor they
return drives span_utils' per-vendor branching (set_model_message_span_attributes,
set_model_choice_span_attributes, _set_finish_reasons_unconditionally), so a wrong
vendor silently drops prompt/completion/finish-reason attributes.

The AWS "global." cross-region inference profile prefix (currently Claude Sonnet 4)
must be recognized alongside the regional prefixes (us / us-gov / eu / apac):
https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html#inference-profiles-support-system
"""

from opentelemetry.instrumentation.bedrock import (
    _cross_region_check,
    _get_vendor_model,
)


class TestCrossRegionCheck:
    """_cross_region_check strips the region prefix and returns (vendor, model)."""

    def test_global_prefix_strips_to_vendor_and_model(self):
        # global. is a real cross-region inference profile prefix (Claude Sonnet 4).
        vendor, model = _cross_region_check(
            "global.anthropic.claude-sonnet-4-20250514-v1:0"
        )
        assert vendor == "anthropic"
        assert model == "claude-sonnet-4-20250514-v1:0"

    def test_regional_prefixes_strip_to_vendor_and_model(self):
        for prefix in ["us", "us-gov", "eu", "apac"]:
            vendor, model = _cross_region_check(
                f"{prefix}.anthropic.claude-3-7-sonnet-20250219-v1:0"
            )
            assert vendor == "anthropic"
            assert model == "claude-3-7-sonnet-20250219-v1:0"

    def test_non_prefixed_model_id_splits_on_first_dot(self):
        vendor, model = _cross_region_check("anthropic.claude-3-haiku-20240307-v1:0")
        assert vendor == "anthropic"
        assert model == "claude-3-haiku-20240307-v1:0"


class TestGetVendorModelGlobalProfile:
    """_get_vendor_model resolves a global. inference profile to the anthropic vendor."""

    def test_global_inference_profile_id(self):
        provider, model_vendor, model = _get_vendor_model(
            "global.anthropic.claude-sonnet-4-20250514-v1:0"
        )
        assert model_vendor == "anthropic"
        assert model == "claude-sonnet-4-20250514-v1:0"

    def test_global_inference_profile_arn(self):
        # The ARN form splits on ":" first, so the trailing version suffix (":0")
        # is parsed out before _cross_region_check sees it (same as the regional
        # ARN form). The load-bearing assertion is that the vendor resolves to
        # "anthropic" and not "global".
        arn = (
            "arn:aws:bedrock:us-east-1:012345678901:inference-profile/"
            "global.anthropic.claude-sonnet-4-20250514-v1:0"
        )
        provider, model_vendor, model = _get_vendor_model(arn)
        assert model_vendor == "anthropic"
        assert model.startswith("claude-sonnet-4-")
