import pytest
import os
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.datasets.datasets import Datasets


@pytest.fixture
def datasets():
    """Create a Datasets instance with HTTP client for VCR recording/playback"""
    api_key = os.environ.get("TRACELOOP_API_KEY", "fake-key-for-vcr-playback")
    base_url = os.environ.get("TRACELOOP_BASE_URL", "https://api-staging.traceloop.com")

    http = HTTPClient(base_url=base_url, api_key=api_key, version="1.0.0")
    return Datasets(http)
