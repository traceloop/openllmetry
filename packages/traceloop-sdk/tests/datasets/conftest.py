import pytest
from unittest.mock import Mock
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.datasets.datasets import Datasets


@pytest.fixture
def mock_http():
    """Create a mock HTTP client"""
    http = Mock(spec=HTTPClient)
    http.post.return_value = {"status": "success"}
    return http


@pytest.fixture
def datasets(mock_http):
    """Create a Datasets instance with mock HTTP client"""
    return Datasets(mock_http)
