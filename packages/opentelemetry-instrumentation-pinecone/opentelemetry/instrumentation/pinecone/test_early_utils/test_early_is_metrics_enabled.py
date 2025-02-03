import os
import pytest
from opentelemetry.instrumentation.pinecone.utils import is_metrics_enabled

@pytest.mark.describe("is_metrics_enabled function")
class TestIsMetricsEnabled:

    @pytest.mark.happy_path
    def test_metrics_enabled_true(self):
        """
        Test that is_metrics_enabled returns True when TRACELOOP_METRICS_ENABLED is set to 'true'.
        """
        os.environ["TRACELOOP_METRICS_ENABLED"] = "true"
        assert is_metrics_enabled() is True

    @pytest.mark.happy_path
    def test_metrics_enabled_true_case_insensitive(self):
        """
        Test that is_metrics_enabled returns True when TRACELOOP_METRICS_ENABLED is set to 'TRUE' (case insensitive).
        """
        os.environ["TRACELOOP_METRICS_ENABLED"] = "TRUE"
        assert is_metrics_enabled() is True

    @pytest.mark.happy_path
    def test_metrics_enabled_false(self):
        """
        Test that is_metrics_enabled returns False when TRACELOOP_METRICS_ENABLED is set to 'false'.
        """
        os.environ["TRACELOOP_METRICS_ENABLED"] = "false"
        assert is_metrics_enabled() is False

    @pytest.mark.happy_path
    def test_metrics_enabled_false_case_insensitive(self):
        """
        Test that is_metrics_enabled returns False when TRACELOOP_METRICS_ENABLED is set to 'FALSE' (case insensitive).
        """
        os.environ["TRACELOOP_METRICS_ENABLED"] = "FALSE"
        assert is_metrics_enabled() is False

    @pytest.mark.edge_case
    def test_metrics_enabled_unset(self):
        """
        Test that is_metrics_enabled returns True when TRACELOOP_METRICS_ENABLED is not set.
        """
        if "TRACELOOP_METRICS_ENABLED" in os.environ:
            del os.environ["TRACELOOP_METRICS_ENABLED"]
        assert is_metrics_enabled() is True

    @pytest.mark.edge_case
    def test_metrics_enabled_random_string(self):
        """
        Test that is_metrics_enabled returns False when TRACELOOP_METRICS_ENABLED is set to a random string.
        """
        os.environ["TRACELOOP_METRICS_ENABLED"] = "random_string"
        assert is_metrics_enabled() is False

    @pytest.mark.edge_case
    def test_metrics_enabled_numeric_string(self):
        """
        Test that is_metrics_enabled returns False when TRACELOOP_METRICS_ENABLED is set to a numeric string.
        """
        os.environ["TRACELOOP_METRICS_ENABLED"] = "123"
        assert is_metrics_enabled() is False

    @pytest.mark.edge_case
    def test_metrics_enabled_none_string(self):
        """
        Test that is_metrics_enabled returns False when TRACELOOP_METRICS_ENABLED is set to 'None'.
        """
        os.environ["TRACELOOP_METRICS_ENABLED"] = "None"
        assert is_metrics_enabled() is False

# Clean up environment variable after tests
@pytest.fixture(autouse=True)
def cleanup_env():
    yield
    if "TRACELOOP_METRICS_ENABLED" in os.environ:
        del os.environ["TRACELOOP_METRICS_ENABLED"]