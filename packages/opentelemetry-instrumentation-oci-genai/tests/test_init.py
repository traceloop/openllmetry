"""Instrumentor lifecycle, suppression and error handling tests (no network, no cassettes)."""

import os
from unittest.mock import MagicMock, patch

import oci
import pytest
from oci.generative_ai_inference import GenerativeAiInferenceClient, models
from opentelemetry import context as context_api
from opentelemetry.instrumentation.oci_genai import OCIGenAIInstrumentor, _wrap
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, Meters
from opentelemetry.trace.status import StatusCode
from wrapt import FunctionWrapper

CHAT_SPEC = {"method": "chat", "details_kwarg": "chat_details", "operation": "chat"}


def _chat_details(compartment_id, model="meta.llama-3.3-70b-instruct"):
    return models.ChatDetails(
        compartment_id=compartment_id,
        serving_mode=models.OnDemandServingMode(model_id=model),
        chat_request=models.GenericChatRequest(
            api_format="GENERIC",
            messages=[models.UserMessage(content=[models.TextContent(text="Tell me a joke")])],
        ),
    )


class TestInstrumentor:
    def test_instrument_and_uninstrument_wrap_client_methods(self, tracer_provider, meter_provider):
        instrumentor = OCIGenAIInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, meter_provider=meter_provider)
        try:
            for method in ("chat", "generate_text", "embed_text", "rerank_text"):
                assert isinstance(vars(GenerativeAiInferenceClient)[method], FunctionWrapper)
        finally:
            instrumentor.uninstrument()
        for method in ("chat", "generate_text", "embed_text", "rerank_text"):
            assert not isinstance(vars(GenerativeAiInferenceClient)[method], FunctionWrapper)

    def test_metrics_disabled(self, tracer_provider, meter_provider):
        with patch.dict(os.environ, {"TRACELOOP_METRICS_ENABLED": "false"}):
            instrumentor = OCIGenAIInstrumentor()
            instrumentor.instrument(tracer_provider=tracer_provider, meter_provider=meter_provider)
            instrumentor.uninstrument()

    def test_missing_methods_are_skipped(self, tracer_provider, meter_provider):
        instrumentor = OCIGenAIInstrumentor()
        with patch(
            "opentelemetry.instrumentation.oci_genai.wrap_function_wrapper",
            side_effect=AttributeError("no such method"),
        ):
            instrumentor.instrument(tracer_provider=tracer_provider, meter_provider=meter_provider)
        instrumentor.uninstrument()

    def test_conflicting_attribute_flags_raise(self):
        with pytest.raises(TypeError):
            OCIGenAIInstrumentor(use_attributes=True, use_legacy_attributes=True)

    def test_use_legacy_attributes_is_deprecated(self):
        with pytest.warns(DeprecationWarning):
            OCIGenAIInstrumentor(use_legacy_attributes=True)


class TestWrap:
    def test_suppression_keys_skip_span(self):
        tracer = MagicMock()
        wrapped = MagicMock(return_value="result")
        wrapper = _wrap(tracer, None, None, None, CHAT_SPEC)

        for key in (_SUPPRESS_INSTRUMENTATION_KEY, SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY):
            token = context_api.attach(context_api.set_value(key, True))
            try:
                assert wrapper(wrapped, None, [_chat_details("c")], {}) == "result"
            finally:
                context_api.detach(token)
        tracer.start_span.assert_not_called()

    def test_details_passed_as_keyword(self):
        tracer = MagicMock()
        span = MagicMock()
        span.is_recording.return_value = True
        tracer.start_span.return_value = span
        response = MagicMock()
        response.data = None
        wrapped = MagicMock(return_value=response)

        wrapper = _wrap(tracer, None, None, None, CHAT_SPEC)
        assert wrapper(wrapped, None, [], {"chat_details": _chat_details("c", model="m")}) is response
        assert tracer.start_span.call_args.args[0] == "chat m"
        span.end.assert_called_once()


def test_service_error_sets_span_status(instrument_legacy, oci_client, compartment_id, span_exporter, reader):
    error = oci.exceptions.ServiceError(
        status=404, code="NotFound", headers={}, message="The requested API is not available."
    )
    with patch.object(oci_client.base_client, "call_api", side_effect=error):
        with pytest.raises(oci.exceptions.ServiceError):
            oci_client.chat(_chat_details(compartment_id))

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat meta.llama-3.3-70b-instruct"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "ServiceError"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "meta.llama-3.3-70b-instruct"
    assert len([event for event in span.events if event.name == "exception"]) == 1

    metrics = reader.get_metrics_data().resource_metrics[0].scope_metrics[0].metrics
    duration = next(metric for metric in metrics if metric.name == Meters.LLM_OPERATION_DURATION)
    assert duration.data.data_points[0].attributes["error.type"] == "ServiceError"


def test_streaming_iteration_error_finishes_span(instrument_legacy, oci_client, compartment_id, span_exporter):
    class FailingSSEClient:
        def events(self):
            yield MagicMock(data='{"index":0,"message":{"role":"ASSISTANT","content":[{"type":"TEXT","text":"Hi"}]}}')
            raise RuntimeError("Mid-stream failure")

    details = _chat_details(compartment_id)
    details.chat_request.is_stream = True
    response = oci.response.Response(200, {}, FailingSSEClient(), None)
    with patch.object(oci_client.base_client, "call_api", return_value=response):
        result = oci_client.chat(details)

    assert span_exporter.get_finished_spans() == ()
    with pytest.raises(RuntimeError, match="Mid-stream failure"):
        for _ in result.data.events():
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].attributes["error.type"] == "RuntimeError"
    assert len([event for event in spans[0].events if event.name == "exception"]) == 1
