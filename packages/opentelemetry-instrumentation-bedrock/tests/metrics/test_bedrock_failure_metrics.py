"""Regression tests for issue #4590 (Bedrock part): a failed call must
record one ``gen_ai.client.operation.duration`` point carrying ``error.type``.

The instrumentors used to record the duration only on the success path (via
``_handle_call`` and the stream-done callbacks), so failed calls never showed
up in error-rate and latency views built on that metric. These tests drive the
real instrumented boto3 client with ``botocore.stub.Stubber`` raising through
``inject_credentials``, so no network is needed.
"""

import json

import pytest
from botocore.stub import Stubber
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE


BODY = json.dumps({"inputText": "hi", "textGenerationConfig": {"maxTokenCount": 10}})


def _duration_points(reader):
    metrics_data = reader.get_metrics_data()
    points = []
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "gen_ai.client.operation.duration":
                    points.extend(metric.data.data_points)
    return points


def test_invoke_model_failure_records_duration_with_error_type(
    instrument_legacy, reader
):
    brt = __import__("boto3").client(
        service_name="bedrock-runtime",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )
    stubber = Stubber(brt)
    stubber.add_client_error(
        "invoke_model",
        service_error_code="AccessDeniedException",
        http_status_code=403,
    )
    stubber.activate()

    with pytest.raises(Exception):
        brt.invoke_model(
            body=BODY,
            modelId="amazon.titan-text-express-v1",
            accept="application/json",
            contentType="application/json",
        )

    stubber.deactivate()

    points = _duration_points(reader)
    assert len(points) == 1
    assert points[0].attributes.get(ERROR_TYPE) == "ClientError"


def test_invoke_model_success_records_duration_without_error_type(
    instrument_legacy, reader
):
    brt = __import__("boto3").client(
        service_name="bedrock-runtime",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )
    stubber = Stubber(brt)
    stubber.add_response(
        "invoke_model",
        expected_params={
            "body": BODY,
            "modelId": "amazon.titan-text-express-v1",
            "accept": "application/json",
            "contentType": "application/json",
        },
        service_response={
            "body": __import__("io").BytesIO(
                json.dumps(
                    {
                        "results": [{"outputText": "ok", "tokenCount": 1}],
                        "outputText": "ok",
                        "tokenCount": 1,
                    }
                ).encode()
            ),
            "contentType": "application/json",
            "ResponseMetadata": {"HTTPHeaders": {}},
        },
    )
    stubber.activate()

    brt.invoke_model(
        body=BODY,
        modelId="amazon.titan-text-express-v1",
        accept="application/json",
        contentType="application/json",
    )
    stubber.deactivate()

    points = _duration_points(reader)
    assert len(points) == 1
    assert ERROR_TYPE not in points[0].attributes
