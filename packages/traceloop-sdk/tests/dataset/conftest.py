import pytest
import os
from unittest.mock import Mock
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.datasets.datasets import Datasets


@pytest.fixture
def datasets():
    """Create a Datasets instance with real HTTP client for VCR recording"""
    # Use staging environment for recording real API calls
    api_key = os.environ.get("TRACELOOP_API_KEY", "fake-key-for-vcr-playback")
    base_url = os.environ.get("TRACELOOP_BASE_URL", "https://api-staging.traceloop.com")
    http = HTTPClient(base_url=base_url, api_key=api_key, version="1.0.0")
    return Datasets(http)


@pytest.fixture
def mock_http():
    """Create a mock HTTP client for tests that still need mocking"""
    http = Mock(spec=HTTPClient)
    http.post.return_value = {"status": "success"}
    http.get.return_value = {"status": "success"}
    http.put.return_value = {"status": "success"}
    http.delete.return_value = True
    return http