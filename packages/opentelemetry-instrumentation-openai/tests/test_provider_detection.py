from opentelemetry.instrumentation.openai.shared import _get_vendor_from_url


def test_detects_atlascloud_provider_from_base_url():
    assert _get_vendor_from_url("https://api.atlascloud.ai/v1") == "atlascloud"


def test_falls_back_to_openai_for_unknown_openai_compatible_provider():
    assert _get_vendor_from_url("https://example.com/v1") == "openai"
