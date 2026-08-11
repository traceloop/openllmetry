from unittest.mock import Mock, patch

import pytest
import requests

from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.datasets.dataset import Dataset
from traceloop.sdk.experiment.experiment import Experiment


def _http_client() -> HTTPClient:
    return HTTPClient(base_url="https://api.example.com", api_key="test-key", version="1.0.0")


def test_http_client_post_returns_json_on_success():
    client = _http_client()
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"status": "ok"}

    with patch("traceloop.sdk.client.http.requests.post", return_value=mock_response):
        result = client.post("annotations", {"k": "v"})

    assert result == {"status": "ok"}


def test_http_client_post_returns_none_on_http_error():
    client = _http_client()
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "500 Server Error"
    )

    with patch("traceloop.sdk.client.http.requests.post", return_value=mock_response):
        result = client.post("annotations", {"k": "v"})

    assert result is None


def test_http_client_post_returns_none_on_transport_error():
    client = _http_client()

    with patch(
        "traceloop.sdk.client.http.requests.post",
        side_effect=requests.exceptions.ConnectionError("connection refused"),
    ):
        result = client.post("annotations", {"k": "v"})

    assert result is None


def test_http_client_post_raises_http_error_with_opt_in():
    client = _http_client()
    mock_response = Mock()
    http_error = requests.exceptions.HTTPError("500 Server Error")
    mock_response.raise_for_status.side_effect = http_error

    with patch("traceloop.sdk.client.http.requests.post", return_value=mock_response):
        with pytest.raises(requests.exceptions.HTTPError):
            client.post("annotations", {"k": "v"}, raise_on_error=True)


def test_http_client_post_raises_transport_error_with_opt_in():
    client = _http_client()
    connection_error = requests.exceptions.ConnectionError("connection refused")

    with patch(
        "traceloop.sdk.client.http.requests.post",
        side_effect=connection_error,
    ):
        with pytest.raises(requests.exceptions.ConnectionError):
            client.post("annotations", {"k": "v"}, raise_on_error=True)


def test_dataset_publish_failure_handling_remains_compatible():
    mock_http = Mock(spec=HTTPClient)
    mock_http.post.return_value = None

    dataset = Dataset(http=mock_http)
    dataset.slug = "test-dataset"

    with pytest.raises(Exception, match="Failed to publish dataset test-dataset"):
        dataset.publish()


def test_experiment_create_task_failure_handling_remains_compatible():
    mock_http_client = Mock(spec=HTTPClient)
    mock_http_client.base_url = "https://api.example.com"
    mock_http_client.post.return_value = None
    mock_async_http_client = Mock()
    experiment = Experiment(mock_http_client, mock_async_http_client, "test-experiment")

    with pytest.raises(Exception, match="Failed to create task for experiment 'test-experiment'"):
        experiment._create_task("test-experiment", "run-123", {}, {"output": "value"})
