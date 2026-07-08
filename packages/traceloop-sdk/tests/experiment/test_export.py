import pytest
from unittest.mock import Mock
from traceloop.sdk.experiment.experiment import Experiment
from traceloop.sdk.client.http import HTTPClient


class TestExportMethods:
    """Tests for to_csv_string() and to_json_string() export methods"""

    def test_to_csv_with_explicit_params(self):
        """Test to_csv_string with explicit experiment_slug and run_id"""
        mock_http_client = Mock(spec=HTTPClient)
        mock_http_client.base_url = "https://api.example.com"
        mock_http_client.get = Mock(return_value="col1,col2\nval1,val2")
        mock_async_http_client = Mock()

        experiment = Experiment(mock_http_client, mock_async_http_client, "test-exp")

        result = experiment.to_csv_string(experiment_slug="my-exp", run_id="run-123")

        assert result == "col1,col2\nval1,val2"
        mock_http_client.get.assert_called_once_with(
            "/experiments/my-exp/runs/run-123/export/csv"
        )

    def test_to_json_with_explicit_params(self):
        """Test to_json_string with explicit experiment_slug and run_id"""
        mock_http_client = Mock(spec=HTTPClient)
        mock_http_client.base_url = "https://api.example.com"
        mock_http_client.get = Mock(return_value={"results": [{"score": 0.9}]})
        mock_async_http_client = Mock()

        experiment = Experiment(mock_http_client, mock_async_http_client, "test-exp")

        result = experiment.to_json_string(experiment_slug="my-exp", run_id="run-123")

        assert result == '{"results": [{"score": 0.9}]}'
        mock_http_client.get.assert_called_once_with(
            "/experiments/my-exp/runs/run-123/export/json"
        )

    def test_to_json_with_string_response(self):
        """Test to_json_string when API returns a string instead of dict"""
        mock_http_client = Mock(spec=HTTPClient)
        mock_http_client.base_url = "https://api.example.com"
        mock_http_client.get = Mock(return_value='{"already": "json"}')
        mock_async_http_client = Mock()

        experiment = Experiment(mock_http_client, mock_async_http_client, "test-exp")

        result = experiment.to_json_string(experiment_slug="my-exp", run_id="run-123")

        assert result == '{"already": "json"}'

    def test_to_csv_uses_last_run_ids(self):
        """Test to_csv_string uses stored IDs from last run"""
        mock_http_client = Mock(spec=HTTPClient)
        mock_http_client.base_url = "https://api.example.com"
        mock_http_client.get = Mock(return_value="csv,data")
        mock_async_http_client = Mock()

        experiment = Experiment(mock_http_client, mock_async_http_client, "")
        experiment._last_experiment_slug = "my-exp"
        experiment._last_run_id = "run-789"

        result = experiment.to_csv_string()

        assert result == "csv,data"
        mock_http_client.get.assert_called_once_with(
            "/experiments/my-exp/runs/run-789/export/csv"
        )

    def test_to_json_uses_last_run_ids(self):
        """Test to_json_string uses stored IDs from last run"""
        mock_http_client = Mock(spec=HTTPClient)
        mock_http_client.base_url = "https://api.example.com"
        mock_http_client.get = Mock(return_value={"data": "value"})
        mock_async_http_client = Mock()

        experiment = Experiment(mock_http_client, mock_async_http_client, "")
        experiment._last_experiment_slug = "my-exp"
        experiment._last_run_id = "run-789"

        result = experiment.to_json_string()

        assert result == '{"data": "value"}'
        mock_http_client.get.assert_called_once_with(
            "/experiments/my-exp/runs/run-789/export/json"
        )

    def test_to_csv_raises_when_no_experiment_slug(self):
        """Test to_csv_string raises ValueError when no experiment_slug available"""
        mock_http_client = Mock(spec=HTTPClient)
        mock_http_client.base_url = "https://api.example.com"
        mock_async_http_client = Mock()

        # Create experiment with empty slug and no last_experiment_slug
        experiment = Experiment(mock_http_client, mock_async_http_client, "")
        experiment._last_experiment_slug = None

        with pytest.raises(ValueError) as exc_info:
            experiment.to_csv_string(run_id="run-123")

        assert "experiment_slug is required" in str(exc_info.value)

    def test_to_csv_raises_when_no_run_id(self):
        """Test to_csv_string raises ValueError when no run_id available"""
        mock_http_client = Mock(spec=HTTPClient)
        mock_http_client.base_url = "https://api.example.com"
        mock_async_http_client = Mock()

        experiment = Experiment(mock_http_client, mock_async_http_client, "")
        experiment._last_experiment_slug = "test-exp"

        with pytest.raises(ValueError) as exc_info:
            experiment.to_csv_string()

        assert "run_id is required" in str(exc_info.value)

    def test_to_json_raises_when_no_experiment_slug(self):
        """Test to_json_string raises ValueError when no experiment_slug available"""
        mock_http_client = Mock(spec=HTTPClient)
        mock_http_client.base_url = "https://api.example.com"
        mock_async_http_client = Mock()

        # Create experiment with empty slug and no last_experiment_slug
        experiment = Experiment(mock_http_client, mock_async_http_client, "")
        experiment._last_experiment_slug = None

        with pytest.raises(ValueError) as exc_info:
            experiment.to_json_string(run_id="run-123")

        assert "experiment_slug is required" in str(exc_info.value)

    def test_to_json_raises_when_no_run_id(self):
        """Test to_json_string raises ValueError when no run_id available"""
        mock_http_client = Mock(spec=HTTPClient)
        mock_http_client.base_url = "https://api.example.com"
        mock_async_http_client = Mock()

        experiment = Experiment(mock_http_client, mock_async_http_client, "")
        experiment._last_experiment_slug = "test-exp"

        with pytest.raises(ValueError) as exc_info:
            experiment.to_json_string()

        assert "run_id is required" in str(exc_info.value)

    def test_to_csv_raises_on_api_failure(self):
        """Test to_csv_string raises Exception when API returns None"""
        mock_http_client = Mock(spec=HTTPClient)
        mock_http_client.base_url = "https://api.example.com"
        mock_http_client.get = Mock(return_value=None)
        mock_async_http_client = Mock()

        experiment = Experiment(mock_http_client, mock_async_http_client, "test-exp")

        with pytest.raises(Exception) as exc_info:
            experiment.to_csv_string(experiment_slug="my-exp", run_id="run-123")

        assert "Failed to export CSV" in str(exc_info.value)

    def test_to_json_raises_on_api_failure(self):
        """Test to_json_string raises Exception when API returns None"""
        mock_http_client = Mock(spec=HTTPClient)
        mock_http_client.base_url = "https://api.example.com"
        mock_http_client.get = Mock(return_value=None)
        mock_async_http_client = Mock()

        experiment = Experiment(mock_http_client, mock_async_http_client, "test-exp")

        with pytest.raises(Exception) as exc_info:
            experiment.to_json_string(experiment_slug="my-exp", run_id="run-123")

        assert "Failed to export JSON" in str(exc_info.value)
