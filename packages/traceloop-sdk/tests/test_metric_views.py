"""Tests for the histogram views returned by ``metric_views()``."""

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.semconv_ai import Meters
from traceloop.sdk.metrics.metrics import metric_views

# Bucket boundaries recommended by the GenAI semantic conventions for
# gen_ai.client.operation.duration (seconds).
OPERATION_DURATION_BOUNDARIES = (
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
)

# Bucket boundaries recommended by the GenAI semantic conventions for
# gen_ai.client.token.usage.
TOKEN_USAGE_BOUNDARIES = (
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
)


def _collect_bounds(instrument_name, unit, value):
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], views=metric_views())
    histogram = provider.get_meter("test").create_histogram(
        instrument_name, unit=unit
    )
    histogram.record(value)

    data = reader.get_metrics_data()
    metrics = [
        metric
        for resource_metrics in data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
        if metric.name == instrument_name
    ]
    assert len(metrics) == 1
    return tuple(metrics[0].data.data_points[0].explicit_bounds)


def test_operation_duration_uses_semconv_second_buckets():
    bounds = _collect_bounds(Meters.LLM_OPERATION_DURATION, "s", 1.5)
    assert bounds == OPERATION_DURATION_BOUNDARIES


def test_token_usage_uses_semconv_token_buckets():
    bounds = _collect_bounds(Meters.LLM_TOKEN_USAGE, "{token}", 100)
    assert bounds == TOKEN_USAGE_BOUNDARIES
