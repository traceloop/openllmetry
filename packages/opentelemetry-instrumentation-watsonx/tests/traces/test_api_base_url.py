from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from opentelemetry.instrumentation.watsonx import (
    WatsonxSpanAttributes,
    _get_api_base_url,
    _handle_input,
    _set_api_attributes,
)


def test_get_api_base_url_from_current_sdk_credentials():
    instance = SimpleNamespace(
        _client=SimpleNamespace(credentials=SimpleNamespace(url="https://eu-de.ml.cloud.ibm.com"))
    )

    assert _get_api_base_url(instance) == "https://eu-de.ml.cloud.ibm.com"


def test_get_api_base_url_from_legacy_sdk_credentials():
    instance = SimpleNamespace(_client=SimpleNamespace(wml_credentials={"url": "https://watsonx.example.internal"}))

    assert _get_api_base_url(instance) == "https://watsonx.example.internal"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"credentials": {"url": "https://jp-tok.ml.cloud.ibm.com"}},
            "https://jp-tok.ml.cloud.ibm.com",
        ),
        (
            {"api_client": SimpleNamespace(credentials=SimpleNamespace(url="https://watsonx.custom"))},
            "https://watsonx.custom",
        ),
    ],
)
def test_get_api_base_url_from_constructor_arguments(kwargs, expected):
    assert _get_api_base_url(kwargs=kwargs) == expected


def test_get_api_base_url_returns_none_when_properties_raise():
    class BrokenClient:
        @property
        def credentials(self):
            raise RuntimeError("credentials unavailable")

        @property
        def wml_credentials(self):
            raise RuntimeError("legacy credentials unavailable")

    class BrokenInstance:
        _client = BrokenClient()

        @property
        def url(self):
            raise RuntimeError("URL unavailable")

        @property
        def credentials(self):
            raise RuntimeError("credentials unavailable")

        @property
        def wml_credentials(self):
            raise RuntimeError("legacy credentials unavailable")

    assert _get_api_base_url(BrokenInstance()) is None


def test_get_api_base_url_returns_none_when_credentials_are_missing():
    assert _get_api_base_url(SimpleNamespace()) is None


@pytest.mark.parametrize("url", [None, "", object()])
def test_get_api_base_url_returns_none_for_invalid_urls(url):
    instance = SimpleNamespace(_client=SimpleNamespace(credentials=SimpleNamespace(url=url)))

    assert _get_api_base_url(instance) is None


def test_set_api_attributes_uses_instance_endpoint():
    span = Mock()
    span.is_recording.return_value = True
    instance = SimpleNamespace(
        _client=SimpleNamespace(credentials=SimpleNamespace(url="https://eu-de.ml.cloud.ibm.com"))
    )

    _set_api_attributes(span, instance)

    span.set_attribute.assert_any_call(
        WatsonxSpanAttributes.WATSONX_API_BASE,
        "https://eu-de.ml.cloud.ibm.com",
    )


def test_set_api_attributes_omits_api_base_when_endpoint_is_unavailable():
    span = Mock()
    span.is_recording.return_value = True

    _set_api_attributes(span, SimpleNamespace())

    assert all(
        call.args[0] != WatsonxSpanAttributes.WATSONX_API_BASE
        for call in span.set_attribute.call_args_list
    )


def test_handle_input_sets_api_base_without_swallowing_signature_error():
    span = Mock()
    span.is_recording.return_value = True
    instance = SimpleNamespace(
        model_id="ibm/granite",
        params=None,
        _client=SimpleNamespace(wml_credentials={"url": "https://watsonx.example.internal"}),
    )

    _handle_input(span, None, "watsonx.generate", instance, (), {})

    span.set_attribute.assert_any_call(
        WatsonxSpanAttributes.WATSONX_API_BASE,
        "https://watsonx.example.internal",
    )
