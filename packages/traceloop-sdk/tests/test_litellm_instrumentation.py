from unittest.mock import patch

import pytest


@pytest.mark.safety
def test_init_litellm_instrumentor_instruments_when_package_is_present():
    from traceloop.sdk.tracing.tracing import init_litellm_instrumentor

    with patch("traceloop.sdk.tracing.tracing.is_package_installed", return_value=True), patch(
        "opentelemetry.instrumentation.litellm.LiteLLMInstrumentor"
    ) as instrumentor_cls:
        instrumentor = instrumentor_cls.return_value
        instrumentor.is_instrumented_by_opentelemetry = False

        assert init_litellm_instrumentor() is True
        instrumentor.instrument.assert_called_once_with()


@pytest.mark.safety
def test_init_litellm_instrumentor_returns_false_when_package_is_missing():
    from traceloop.sdk.tracing.tracing import init_litellm_instrumentor

    with patch("traceloop.sdk.tracing.tracing.is_package_installed", return_value=False):
        assert init_litellm_instrumentor() is False


@pytest.mark.safety
def test_init_instrumentations_dispatches_litellm():
    from traceloop.sdk.instruments import Instruments
    from traceloop.sdk.tracing.tracing import init_instrumentations

    with patch(
        "traceloop.sdk.tracing.tracing.init_litellm_instrumentor", return_value=True
    ) as init_mock:
        init_instrumentations(
            should_enrich_metrics=False,
            base64_image_uploader=lambda *args, **kwargs: "",
            instruments={Instruments.LITELLM},
        )

    init_mock.assert_called_once_with()
