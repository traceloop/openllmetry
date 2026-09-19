"""Regression tests for #4419: streaming responses must record token and
duration metrics.

The streaming path (_create_stream_processor) collected usage from each chunk
but never recorded it to the token/duration histograms, so streaming calls were
invisible on metric dashboards. These tests drive the processor with mocked
chunks and assert the metrics land.
"""

from unittest.mock import MagicMock

from opentelemetry.instrumentation.groq import _create_stream_processor
from opentelemetry.semconv_ai import Meters


def _chunk(*, prompt_tokens=0, completion_tokens=0):
    """Mock a Groq streaming chunk carrying x_groq.usage (as the final chunk does)."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = ""
    chunk.choices[0].delta.tool_calls = None
    chunk.choices[0].finish_reason = None
    chunk.x_groq = MagicMock()
    chunk.x_groq.usage.prompt_tokens = prompt_tokens
    chunk.x_groq.usage.completion_tokens = completion_tokens
    chunk.x_groq.usage.total_tokens = prompt_tokens + completion_tokens
    return chunk


def _collect_metric_names(reader):
    data = reader.get_metrics_data()
    names = set()
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                names.add(metric.name)
    return names


def test_streaming_records_token_and_duration_metrics(reader, meter_provider):
    """Streaming calls record LLM_TOKEN_USAGE and LLM_OPERATION_DURATION (fixes #4419)."""

    meter = meter_provider.get_meter("groq-streaming")
    token_histogram = meter.create_histogram(name=Meters.LLM_TOKEN_USAGE, unit="token")
    duration_histogram = meter.create_histogram(name=Meters.LLM_OPERATION_DURATION, unit="s")

    span = MagicMock()
    span.is_recording.return_value = False

    resp = [_chunk(), _chunk(prompt_tokens=9, completion_tokens=4)]
    gen = _create_stream_processor(resp, span, None, token_histogram, duration_histogram, 0.0)
    for _ in gen:
        pass

    names = _collect_metric_names(reader)
    assert Meters.LLM_TOKEN_USAGE in names, f"token usage not recorded, got {names}"
    assert Meters.LLM_OPERATION_DURATION in names, f"operation duration not recorded, got {names}"
