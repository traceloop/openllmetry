"""Unit tests for dynamic API base URL extraction from Watsonx instances."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from opentelemetry.instrumentation.watsonx import WatsonxSpanAttributes
from opentelemetry.instrumentation.watsonx import (
    _get_api_base_url,
    _set_api_attributes,
)


DEFAULT_URL = "https://us-south.ml.cloud.ibm.com"
FRANKFURT_URL = "https://eu-de.ml.cloud.ibm.com"
TOKYO_URL = "https://jp-tok.ml.cloud.ibm.com"


class TestGetApiBaseUrl:
    """Tests for _get_api_base_url helper."""

    def test_returns_url_from_instance_credentials(self):
        instance = SimpleNamespace(
            _client=SimpleNamespace(
                credentials=SimpleNamespace(url=FRANKFURT_URL)
            )
        )
        assert _get_api_base_url(instance) == FRANKFURT_URL

    def test_returns_default_when_instance_is_none(self):
        assert _get_api_base_url(None) == DEFAULT_URL

    def test_returns_default_when_client_missing(self):
        instance = SimpleNamespace()
        assert _get_api_base_url(instance) == DEFAULT_URL

    def test_returns_default_when_credentials_missing(self):
        instance = SimpleNamespace(_client=SimpleNamespace())
        assert _get_api_base_url(instance) == DEFAULT_URL

    def test_returns_default_when_url_is_none(self):
        instance = SimpleNamespace(
            _client=SimpleNamespace(
                credentials=SimpleNamespace(url=None)
            )
        )
        assert _get_api_base_url(instance) == DEFAULT_URL

    def test_returns_default_when_url_is_empty(self):
        instance = SimpleNamespace(
            _client=SimpleNamespace(
                credentials=SimpleNamespace(url="")
            )
        )
        assert _get_api_base_url(instance) == DEFAULT_URL

    def test_handles_different_regions(self):
        instance = SimpleNamespace(
            _client=SimpleNamespace(
                credentials=SimpleNamespace(url=TOKYO_URL)
            )
        )
        assert _get_api_base_url(instance) == TOKYO_URL

    def test_handles_attribute_error_gracefully(self):
        """Ensure no exception is raised when accessing attributes fails."""
        instance = MagicMock()
        instance._client.credentials = property(
            lambda self: (_ for _ in ()).throw(AttributeError)
        )
        # Should fall back to default, not raise
        result = _get_api_base_url(instance)
        assert isinstance(result, str)


class TestSetApiAttributes:
    """Tests for _set_api_attributes with dynamic URL."""

    def test_sets_url_from_instance(self):
        span = MagicMock()
        span.is_recording.return_value = True
        instance = SimpleNamespace(
            _client=SimpleNamespace(
                credentials=SimpleNamespace(url=FRANKFURT_URL)
            )
        )

        _set_api_attributes(span, instance)

        span.set_attribute.assert_any_call(
            WatsonxSpanAttributes.WATSONX_API_BASE, FRANKFURT_URL
        )

    def test_sets_default_url_when_no_instance(self):
        span = MagicMock()
        span.is_recording.return_value = True

        _set_api_attributes(span)

        span.set_attribute.assert_any_call(
            WatsonxSpanAttributes.WATSONX_API_BASE, DEFAULT_URL
        )

    def test_sets_api_type_and_version(self):
        span = MagicMock()
        span.is_recording.return_value = True

        _set_api_attributes(span)

        span.set_attribute.assert_any_call(
            WatsonxSpanAttributes.WATSONX_API_TYPE, "watsonx.ai"
        )
        span.set_attribute.assert_any_call(
            WatsonxSpanAttributes.WATSONX_API_VERSION, "1.0"
        )

    def test_skips_when_not_recording(self):
        span = MagicMock()
        span.is_recording.return_value = False

        _set_api_attributes(span)

        span.set_attribute.assert_not_called()
