import ollama
import pytest
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics as GenAIMetrics
from opentelemetry.semconv_ai import Meters, SpanAttributes


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
    # Assert metrics for token usage, operation duration, and time to first token are present
    assert any(
        name == Meters.LLM_TOKEN_USAGE for name, _ in points
    ), "Token usage metric not found"
    assert any(
        name == Meters.LLM_OPERATION_DURATION for name, _ in points
    ), "Operation duration metric not found"
    assert any(
        name == GenAIMetrics.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN for name, _ in points
    ), "Time to first token metric not found"

    # Further assert that time-to-first-token is greater than 0 and has the system attribute
    for name, dp in points:
        if name == GenAIMetrics.GEN_AI_SERVER_TIME_TO_FIRST_TOKEN:
            assert dp.sum > 0, "Time to first token should be greater than 0"
            assert dp.attributes.get(SpanAttributes.LLM_SYSTEM) == "Ollama"
            break
