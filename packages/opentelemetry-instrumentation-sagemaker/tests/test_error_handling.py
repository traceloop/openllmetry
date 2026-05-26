from unittest.mock import patch
from botocore.exceptions import ClientError

import pytest
from opentelemetry.trace.status import StatusCode


def _make_client_error(operation="InvokeEndpoint"):
    return ClientError(
        {"Error": {"Code": "ModelError", "Message": "Model invocation failed"}},
        operation,
    )


def test_sagemaker_invoke_error_sets_span_status(instrument_legacy, smrt, span_exporter):
    with patch("botocore.endpoint.URLLib3Session.send", side_effect=_make_client_error()):
        with pytest.raises(ClientError):
            smrt.invoke_endpoint(
                EndpointName="my-test-endpoint",
                Body=b'{"inputs": "Tell me a joke"}',
                ContentType="application/json",
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") == "ClientError"
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1
    assert "ModelError" in events[0].attributes["exception.message"]


def test_sagemaker_invoke_stream_error_sets_span_status(instrument_legacy, smrt, span_exporter):
    with patch(
        "botocore.endpoint.URLLib3Session.send",
        side_effect=_make_client_error("InvokeEndpointWithResponseStream"),
    ):
        with pytest.raises(ClientError):
            smrt.invoke_endpoint_with_response_stream(
                EndpointName="my-test-endpoint",
                Body=b'{"inputs": "Tell me a joke"}',
                ContentType="application/json",
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") == "ClientError"
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1
    assert "ModelError" in events[0].attributes["exception.message"]
