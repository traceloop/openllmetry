import json

import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from opentelemetry.instrumentation.bedrock import GuardrailMeters


@pytest.mark.vcr
def test_titan_invoke_model_guardrail(test_context, brt):
    _, _, reader = test_context

    body = json.dumps({})

    brt.invoke_model(
        body=body,
        modelId="amazon.titan-text-express-v1",
    )

    assert_guardrails(reader)


@pytest.mark.vcr
def test_titan_invoke_stream_guardrail(test_context, brt):
    _, _, reader = test_context

    body = json.dumps({})

    r = brt.invoke_model_with_response_stream(
        body=body,
        modelId="amazon.titan-text-express-v1",
    )
    # consume the stream to observe it
    for _ in r.get("body"):
        continue
    assert_guardrails(reader)


@pytest.mark.vcr
def test_titan_converse_guardrail(test_context, brt):
    _, _, reader = test_context

    guardrail = {
        "guardrailIdentifier": "5zwrmdlsra2e",
        "guardrailVersion": "DRAFT",
        "trace": "enabled",
    }

    brt.converse(
        modelId="amazon.titan-text-express-v1",
        messages=[],
        guardrailConfig=guardrail,
    )

    assert_guardrails(reader)


@pytest.mark.vcr
def test_titan_converse_stream_guardrail(test_context, brt):
    _, _, reader = test_context

    guardrail = {
        "guardrailIdentifier": "5zwrmdlsra2e",
        "guardrailVersion": "DRAFT",
        "trace": "enabled",
    }

    r = brt.converse_stream(
        modelId="amazon.titan-text-express-v1",
        messages=[],
        guardrailConfig=guardrail,
    )

    # consume the stream to observe it
    for _ in r.get("stream"):
        continue

    assert_guardrails(reader)


def assert_guardrails(reader):

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_activations = False
    found_latency = False
    found_coverage = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:

                if metric.name == GuardrailMeters.LLM_BEDROCK_GUARDRAIL_ACTIVATION:
                    found_activations = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes["gen_ai.guardrail"] != ""
                        assert data_point.value > 0

                if metric.name == GuardrailMeters.LLM_BEDROCK_GUARDRAIL_LATENCY:
                    found_latency = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )

                if metric.name == GuardrailMeters.LLM_BEDROCK_GUARDRAIL_COVERAGE:
                    found_coverage = True
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert data_point.value > 0

                assert (
                    metric.data.data_points[0].attributes[GenAIAttributes.GEN_AI_SYSTEM]
                    == "bedrock"
                )

    assert found_activations is True
    assert found_latency is True
    assert found_coverage is True
