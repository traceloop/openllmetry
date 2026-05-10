from unittest.mock import patch
from botocore.exceptions import ClientError

import pytest
from opentelemetry.trace.status import StatusCode


def _make_client_error(operation="InvokeModel"):
    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        operation,
    )


def test_bedrock_invoke_model_error_sets_span_status(instrument_legacy, brt, span_exporter):
    with patch("botocore.endpoint.URLLib3Session.send", side_effect=_make_client_error("InvokeModel")):
        with pytest.raises(ClientError):
            brt.invoke_model(
                body=b'{"prompt": "test", "max_tokens_to_sample": 10}',
                modelId="anthropic.claude-v2:1",
                accept="application/json",
                contentType="application/json",
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1


def test_bedrock_converse_error_sets_span_status(instrument_legacy, brt, span_exporter):
    with patch("botocore.endpoint.URLLib3Session.send", side_effect=_make_client_error("Converse")):
        with pytest.raises(ClientError):
            brt.converse(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                messages=[{"role": "user", "content": [{"text": "Tell me a joke"}]}],
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    events = [e for e in span.events if e.name == "exception"]
    assert len(events) == 1
