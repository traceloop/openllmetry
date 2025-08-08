"""Tests for MetricBuckets and MetricViewsBuilder classes."""

from opentelemetry.semconv_ai import MetricBuckets, MetricViewsBuilder, Meters


class TestMetricBuckets:
    """Test the MetricBuckets class and its predefined bucket boundaries."""

    def test_llm_token_usage_buckets(self):
        """Test LLM token usage bucket boundaries."""
        buckets = MetricBuckets.LLM_TOKEN_USAGE

        assert isinstance(buckets, list)
        assert len(buckets) == 14

        assert buckets == [
            1, 4, 16, 64, 256, 1024, 4096, 16384,
            65536, 262144, 1048576, 4194304, 16777216, 67108864
        ]

        for i in range(len(buckets) - 1):
            assert buckets[i] < buckets[i + 1]

    def test_pinecone_query_duration_buckets(self):
        """Test Pinecone query duration bucket boundaries."""
        buckets = MetricBuckets.PINECONE_DB_QUERY_DURATION

        assert isinstance(buckets, list)
        assert len(buckets) == 14

        assert buckets[0] == 0.01
        assert buckets[-1] == 81.92

        for i in range(len(buckets) - 1):
            assert buckets[i] < buckets[i + 1]

    def test_pinecone_query_scores_buckets(self):
        """Test Pinecone query scores bucket boundaries."""
        buckets = MetricBuckets.PINECONE_DB_QUERY_SCORES

        assert isinstance(buckets, list)
        assert len(buckets) == 17

        assert buckets[0] == -1
        assert buckets[-1] == 1
        assert 0 in buckets

        for i in range(len(buckets) - 1):
            assert buckets[i] < buckets[i + 1]


class TestMetricViewsBuilder:
    """Test the MetricViewsBuilder class."""

    def test_get_llm_token_usage_view(self):
        """Test creating LLM token usage view."""
        view = MetricViewsBuilder.get_llm_token_usage_view()

        assert view is not None
        assert view._instrument_name == Meters.LLM_TOKEN_USAGE

        try:
            from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation
            assert isinstance(view._aggregation, ExplicitBucketHistogramAggregation)
            assert view._aggregation._boundaries == MetricBuckets.LLM_TOKEN_USAGE
        except ImportError:
            pass

    def test_get_pinecone_query_duration_view(self):
        """Test creating Pinecone query duration view."""
        view = MetricViewsBuilder.get_pinecone_query_duration_view()

        assert view is not None
        assert view._instrument_name == Meters.PINECONE_DB_QUERY_DURATION

        try:
            from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation
            assert isinstance(view._aggregation, ExplicitBucketHistogramAggregation)
            assert view._aggregation._boundaries == MetricBuckets.PINECONE_DB_QUERY_DURATION
        except ImportError:
            pass

    def test_get_pinecone_query_scores_view(self):
        """Test creating Pinecone query scores view."""
        view = MetricViewsBuilder.get_pinecone_query_scores_view()

        assert view is not None
        assert view._instrument_name == Meters.PINECONE_DB_QUERY_SCORES

        try:
            from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation
            assert isinstance(view._aggregation, ExplicitBucketHistogramAggregation)
            assert view._aggregation._boundaries == MetricBuckets.PINECONE_DB_QUERY_SCORES
        except ImportError:
            pass

    def test_get_llm_operation_duration_view(self):
        """Test creating LLM operation duration view."""
        view = MetricViewsBuilder.get_llm_operation_duration_view()

        assert view is not None
        assert view._instrument_name == Meters.LLM_OPERATION_DURATION

        # LLM operation duration view uses default aggregation (no custom buckets)
        try:
            from opentelemetry.sdk.metrics.view import View
            assert isinstance(view, View)
        except ImportError:
            pass

    def test_get_db_query_duration_view(self):
        """Test creating database query duration view."""
        view = MetricViewsBuilder.get_db_query_duration_view()

        assert view is not None
        assert view._instrument_name == Meters.DB_QUERY_DURATION

        try:
            from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation
            assert isinstance(view._aggregation, ExplicitBucketHistogramAggregation)
            assert view._aggregation._boundaries == MetricBuckets.PINECONE_DB_QUERY_DURATION
        except ImportError:
            pass

    def test_get_db_search_distance_view(self):
        """Test creating database search distance view."""
        view = MetricViewsBuilder.get_db_search_distance_view()

        assert view is not None
        assert view._instrument_name == Meters.DB_SEARCH_DISTANCE

        try:
            from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation
            assert isinstance(view._aggregation, ExplicitBucketHistogramAggregation)
            assert view._aggregation._boundaries == MetricBuckets.PINECONE_DB_QUERY_SCORES
        except ImportError:
            pass

    def test_get_all_genai_views(self):
        """Test getting all GenAI views."""
        views = MetricViewsBuilder.get_all_genai_views()

        assert isinstance(views, list)
        assert len(views) == 6

        view_names = [view._instrument_name for view in views]
        assert Meters.LLM_TOKEN_USAGE in view_names
        assert Meters.PINECONE_DB_QUERY_DURATION in view_names
        assert Meters.PINECONE_DB_QUERY_SCORES in view_names
        assert Meters.LLM_OPERATION_DURATION in view_names
        assert Meters.DB_QUERY_DURATION in view_names
        assert Meters.DB_SEARCH_DISTANCE in view_names
