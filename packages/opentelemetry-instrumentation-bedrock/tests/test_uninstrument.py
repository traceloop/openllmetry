"""Tests for BedrockInstrumentor.uninstrument() on pre-existing clients.

Issue #4577: clients created while the instrumentor is active keep emitting
spans after uninstrument() because _uninstrument() only removes the
create_client factory wrapper and leaves per-client method wrappers in place.
"""

import boto3
import pytest
from botocore.exceptions import ClientError
from unittest.mock import patch

from opentelemetry.instrumentation.bedrock import BedrockInstrumentor


def _throttle_error(operation="InvokeModel"):
    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        operation,
    )


def test_uninstrument_stops_spans_on_existing_invoke_model_client(
    tracer_provider, span_exporter
):
    """Pre-existing client must not create spans after uninstrument()."""
    instrumentor = BedrockInstrumentor(enrich_token_usage=True)
    instrumentor.uninstrument()
    instrumentor.instrument(tracer_provider=tracer_provider)

    client = boto3.client(
        "bedrock-runtime",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )

    instrumentor.uninstrument()

    with patch(
        "botocore.endpoint.URLLib3Session.send",
        side_effect=_throttle_error("InvokeModel"),
    ):
        with pytest.raises(ClientError):
            client.invoke_model(
                body=b'{"prompt": "test", "max_tokens_to_sample": 10}',
                modelId="anthropic.claude-v2:1",
                accept="application/json",
                contentType="application/json",
            )

    assert span_exporter.get_finished_spans() == []


def test_uninstrument_stops_spans_on_existing_converse_client(
    tracer_provider, span_exporter
):
    """Pre-existing client.converse() must not create spans after uninstrument()."""
    instrumentor = BedrockInstrumentor(enrich_token_usage=True)
    instrumentor.uninstrument()
    instrumentor.instrument(tracer_provider=tracer_provider)

    client = boto3.client(
        "bedrock-runtime",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )

    instrumentor.uninstrument()

    with patch(
        "botocore.endpoint.URLLib3Session.send",
        side_effect=_throttle_error("Converse"),
    ):
        with pytest.raises(ClientError):
            client.converse(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            )

    assert span_exporter.get_finished_spans() == []


def test_reinstrument_resumes_spans_after_uninstrument(tracer_provider, span_exporter):
    """Re-instrumenting after uninstrument() should resume span creation."""
    instrumentor = BedrockInstrumentor(enrich_token_usage=True)
    instrumentor.uninstrument()
    instrumentor.instrument(tracer_provider=tracer_provider)
    instrumentor.uninstrument()
    instrumentor.instrument(tracer_provider=tracer_provider)

    client = boto3.client(
        "bedrock-runtime",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )

    with patch(
        "botocore.endpoint.URLLib3Session.send",
        side_effect=_throttle_error("InvokeModel"),
    ):
        with pytest.raises(ClientError):
            client.invoke_model(
                body=b'{"prompt": "test", "max_tokens_to_sample": 10}',
                modelId="anthropic.claude-v2:1",
                accept="application/json",
                contentType="application/json",
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    instrumentor.uninstrument()
