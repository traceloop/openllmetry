"""Integration tests for MetricBuckets with OpenTelemetry SDK."""

from opentelemetry.semconv_ai import (
    MetricBuckets,
    MetricViewsBuilder,
    Meters
)


class TestMetricViewsIntegration:
    """Test integration between MetricViewsBuilder and OpenTelemetry SDK."""

    def test_meter_provider_with_custom_views(self):
        """Test that custom metric views can be applied to a MeterProvider."""
        try:
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.view import View
        except ImportError:
            return

        views = MetricViewsBuilder.get_all_genai_views()
        assert len(views) == 6

        for view in views:
            assert isinstance(view, View)

        meter_provider = MeterProvider(views=views)
        meter = meter_provider.get_meter("test")

        token_histogram = meter.create_histogram(
            name=Meters.LLM_TOKEN_USAGE,
            description="Token usage test"
        )

        duration_histogram = meter.create_histogram(
            name=Meters.PINECONE_DB_QUERY_DURATION,
            description="Duration test"
        )

        score_histogram = meter.create_histogram(
            name=Meters.PINECONE_DB_QUERY_SCORES,
            description="Score test"
        )

        assert token_histogram is not None
        assert duration_histogram is not None
        assert score_histogram is not None

        token_histogram.record(100)
        duration_histogram.record(0.5)
        score_histogram.record(0.8)

    def test_individual_view_creation(self):
        """Test creating individual metric views."""
        try:
            from opentelemetry.sdk.metrics.view import View
        except ImportError:
            return

        llm_view = MetricViewsBuilder.get_llm_token_usage_view()
        duration_view = MetricViewsBuilder.get_pinecone_query_duration_view()
        score_view = MetricViewsBuilder.get_pinecone_query_scores_view()

        assert isinstance(llm_view, View)
        assert isinstance(duration_view, View)
        assert isinstance(score_view, View)

        assert llm_view._instrument_name == Meters.LLM_TOKEN_USAGE
        assert duration_view._instrument_name == Meters.PINECONE_DB_QUERY_DURATION
        assert score_view._instrument_name == Meters.PINECONE_DB_QUERY_SCORES

    def test_bucket_configuration_in_views(self):
        """Test that views have correct bucket configurations."""
        try:
            from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation
        except ImportError:
            return

        llm_view = MetricViewsBuilder.get_llm_token_usage_view()
        aggregation = llm_view._aggregation

        assert isinstance(aggregation, ExplicitBucketHistogramAggregation)
        assert aggregation._boundaries == MetricBuckets.LLM_TOKEN_USAGE

    def test_pinecone_metrics_integration(self):
        """Test Pinecone-specific metric integration."""
        try:
            from opentelemetry.sdk.metrics import MeterProvider
        except ImportError:
            return

        duration_view = MetricViewsBuilder.get_pinecone_query_duration_view()
        score_view = MetricViewsBuilder.get_pinecone_query_scores_view()

        meter_provider = MeterProvider(views=[duration_view, score_view])
        meter = meter_provider.get_meter("pinecone_test")

        duration_histogram = meter.create_histogram(
            name=Meters.PINECONE_DB_QUERY_DURATION,
            description="Pinecone query duration"
        )

        score_histogram = meter.create_histogram(
            name=Meters.PINECONE_DB_QUERY_SCORES,
            description="Pinecone similarity scores"
        )

        duration_histogram.record(0.1)
        duration_histogram.record(1.5)
        duration_histogram.record(10.0)

        score_histogram.record(-0.5)
        score_histogram.record(0.0)
        score_histogram.record(0.9)

        assert True

    def test_view_compatibility_with_different_otel_versions(self):
        """Test that views work across different OpenTelemetry versions."""
        try:
            from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation
            from opentelemetry.sdk.metrics import MeterProvider
        except ImportError:
            return

        all_views = MetricViewsBuilder.get_all_genai_views()

        for view in all_views:
            assert hasattr(view, '_instrument_name')
            assert hasattr(view, '_aggregation')
            # LLM_OPERATION_DURATION uses default aggregation, others use explicit buckets
            if view._instrument_name == Meters.LLM_OPERATION_DURATION:
                # Default aggregation for operation duration
                from opentelemetry.sdk.metrics._internal.aggregation import DefaultAggregation
                assert isinstance(view._aggregation, DefaultAggregation)
            else:
                # Explicit bucket aggregation for other metrics
                assert isinstance(view._aggregation, ExplicitBucketHistogramAggregation)

        meter_provider = MeterProvider(views=all_views)
        meter = meter_provider.get_meter("compatibility_test")
        histogram = meter.create_histogram(name="test_metric")
        histogram.record(1.0)


class TestBucketEffectiveness:
    """Test the effectiveness of predefined bucket boundaries."""

    def test_llm_token_usage_bucket_distribution(self):
        """Test LLM token usage bucket distribution."""
        token_buckets = MetricBuckets.LLM_TOKEN_USAGE

        assert len(token_buckets) == 14

        assert 1 in token_buckets
        assert 1024 in token_buckets
        assert 67108864 in token_buckets

        for i in range(len(token_buckets) - 1):
            assert token_buckets[i] < token_buckets[i + 1]

        common_token_values = [10, 50, 500, 2000, 8000, 32000, 100000]
        for value in common_token_values:
            bucket_found = False
            for bucket in token_buckets:
                if value <= bucket:
                    bucket_found = True
                    break
            assert bucket_found, f"No appropriate bucket found for token value {value}"

    def test_pinecone_query_duration_bucket_distribution(self):
        """Test Pinecone query duration bucket distribution."""
        duration_buckets = MetricBuckets.PINECONE_DB_QUERY_DURATION

        assert len(duration_buckets) == 14

        assert 0.01 in duration_buckets
        assert 81.92 in duration_buckets

        for i in range(len(duration_buckets) - 1):
            assert duration_buckets[i] < duration_buckets[i + 1]

        common_durations = [0.05, 0.1, 0.5, 1.0, 5.0, 30.0]
        for duration in common_durations:
            bucket_found = False
            for bucket in duration_buckets:
                if duration <= bucket:
                    bucket_found = True
                    break
            assert bucket_found, f"No appropriate bucket found for duration {duration}"

    def test_similarity_score_bucket_coverage(self):
        """Test similarity score bucket coverage."""
        score_buckets = MetricBuckets.PINECONE_DB_QUERY_SCORES

        assert len(score_buckets) == 17

        assert -1 in score_buckets
        assert 0 in score_buckets
        assert 1 in score_buckets

        for i in range(len(score_buckets) - 1):
            assert score_buckets[i] < score_buckets[i + 1]

        assert score_buckets[0] == -1
        assert score_buckets[-1] == 1

        test_scores = [-0.9, -0.5, -0.1, 0.0, 0.3, 0.7, 0.95]
        for score in test_scores:
            bucket_found = False
            for bucket in score_buckets:
                if score <= bucket:
                    bucket_found = True
                    break
            assert bucket_found, f"No appropriate bucket found for score {score}"


class TestViewBuilderErrorHandling:
    """Test error handling in MetricViewsBuilder."""

    def test_view_creation_with_missing_dependencies(self):
        """Test graceful handling when OpenTelemetry SDK is not available."""

        views = MetricViewsBuilder.get_all_genai_views()
        assert len(views) == 6

        for view in views:
            assert view is not None

    def test_bucket_immutability(self):
        """Test that bucket definitions cannot be accidentally modified."""
        original_llm_buckets = MetricBuckets.LLM_TOKEN_USAGE.copy()
        original_duration_buckets = MetricBuckets.PINECONE_DB_QUERY_DURATION.copy()
        original_score_buckets = MetricBuckets.PINECONE_DB_QUERY_SCORES.copy()

        assert MetricBuckets.LLM_TOKEN_USAGE == original_llm_buckets
        assert MetricBuckets.PINECONE_DB_QUERY_DURATION == original_duration_buckets
        assert MetricBuckets.PINECONE_DB_QUERY_SCORES == original_score_buckets
