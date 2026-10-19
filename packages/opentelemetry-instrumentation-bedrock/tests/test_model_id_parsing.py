"""Model IDs whose model name contains a dot (openai.gpt-5.6-sol) must survive
the cross-region prefix strip, the same way the prefix-less form does."""

from opentelemetry.instrumentation.bedrock import _get_vendor_model


class TestDottedModelName:
    def test_regional_profile_keeps_full_model_name(self):
        assert _get_vendor_model("us.openai.gpt-5.6-sol") == ("aws.bedrock", "openai", "gpt-5.6-sol")

    def test_regional_profile_arn_keeps_full_model_name(self):
        arn = "arn:aws:bedrock:us-east-1:111122223333:inference-profile/us.openai.gpt-5.6-sol"
        assert _get_vendor_model(arn) == ("aws.bedrock", "openai", "gpt-5.6-sol")

    def test_prefixless_id_unchanged(self):
        assert _get_vendor_model("openai.gpt-5.6-sol") == ("aws.bedrock", "openai", "gpt-5.6-sol")

    def test_undotted_profile_unchanged(self):
        assert _get_vendor_model("us.openai.gpt-6-sol") == ("aws.bedrock", "openai", "gpt-6-sol")
        assert _get_vendor_model("eu.anthropic.claude-3-7-sonnet-20250219-v1:0") == (
            "aws.bedrock",
            "anthropic",
            "claude-3-7-sonnet-20250219-v1:0",
        )
