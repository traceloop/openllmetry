import ollama
import pytest
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.metrics import (
    gen_ai_metrics as GenAIMetrics,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.semconv_ai import Meters


def _collect_metrics(reader):
    """Helper to flatten all metrics data points."""
    data = reader.get_metrics_data()
    points = []
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                for dp in metric.data.data_points:
                    points.append((metric.name, dp))
    return points


@pytest.mark.vcr
def test_ollama_streaming_metrics(instrument_legacy, reader):
    gen = ollama.generate(
        model="gemma3:1b",
        prompt="Tell me a joke about OpenTelemetry",
        stream=True,
    )
    for _ in gen:
        pass

    points = _collect_metrics(reader)
    # Assert metrics for token usage, operation duration, time to first token,
    # and streaming time to generate are present
    assert any(name == Meters.LLM_TOKEN_USAGE for name, _ in points), "Token usage metric not found"
    assert any(name == Meters.LLM_OPERATION_DURATION for name, _ in points), "Operation duration metric not found"
    assert any(name == GenAIMetrics.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN for name, _ in points), \
        "Time to first token metric not found"
    assert any(name == Meters.LLM_STREAMING_TIME_TO_GENERATE for name, _ in points), \
        "Streaming time to generate metric not found"

    # Further assert that time-to-first-token is greater than 0 and has the system attribute
    for name, dp in points:
        if name == GenAIMetrics.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN:
            assert dp.sum > 0, "Time to first token should be greater than 0"
            assert dp.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "Ollama"
            break


@pytest.mark.vcr
def test_ollama_streaming_time_to_generate_metrics(instrument_legacy, reader):
    gen = ollama.generate(
        model="gemma3:1b",
        prompt="Tell me a joke about OpenTelemetry",
        stream=True,
    )
    for _ in gen:
        pass

    points = _collect_metrics(reader)
    # Assert metrics for streaming time to generate is present
    assert any(name == Meters.LLM_STREAMING_TIME_TO_GENERATE for name, _ in points), \
        "Streaming time to generate metric not found"

    # Further assert that streaming-time-to-generate is greater than 0 and has the system attribute
    for name, dp in points:
        if name == Meters.LLM_STREAMING_TIME_TO_GENERATE:
            assert dp.sum > 0, "Streaming time to generate should be greater than 0"
            assert dp.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "Ollama"
            assert dp.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL) is not None
            break


@pytest.mark.vcr
def test_ollama_operation_duration_includes_model_attribute(instrument_legacy, reader):
    """Test that LLM_OPERATION_DURATION metric includes gen_ai.response.model attribute."""
    ollama.chat(
        model="gemma3:1b",
        messages=[
            {"role": "user", "content": "Hello, this is a test for model attribute."},
        ],
    )

    points = _collect_metrics(reader)

    operation_duration_found = False
    model_attribute_found = False

    for name, dp in points:
        if name == Meters.LLM_OPERATION_DURATION:
            operation_duration_found = True
            # Check that the metric has both required attributes
            assert dp.attributes.get(SpanAttributes.LLM_SYSTEM) == "Ollama", \
                "LLM_OPERATION_DURATION should have gen_ai.system attribute"

            model_name = dp.attributes.get(SpanAttributes.LLM_RESPONSE_MODEL)
            if model_name is not None:
                model_attribute_found = True
                assert model_name == "gemma3:1b", \
                    f"Expected model 'gemma3:1b', but got '{model_name}'"

            assert dp.sum > 0, "Operation duration should be greater than 0"
            break

    assert operation_duration_found, "LLM_OPERATION_DURATION metric not found"
    assert model_attribute_found, "gen_ai.response.model attribute not found in LLM_OPERATION_DURATION metric"
