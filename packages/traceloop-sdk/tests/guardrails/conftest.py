"""Fixtures for guardrail integration tests."""

import os
import pytest
import httpx


@pytest.fixture(scope="module")
def vcr_config():
    """VCR configuration for guardrail integration tests.

    Filters authorization headers to avoid storing API keys in cassettes.
    """
    return {
        "filter_headers": ["authorization", "Authorization"],
        "record_mode": "once",
    }


@pytest.fixture
async def async_http_client():
    """Create async HTTP client for evaluator API calls.

    Uses environment variables for recording, fake values for playback.
    """
    api_key = os.environ.get("TRACELOOP_API_KEY", "fake-key-for-vcr-playback")
    base_url = os.environ.get(
        "TRACELOOP_BASE_URL",
        "https://api-staging.traceloop.com"
    )

    client = httpx.AsyncClient(
        base_url=base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(120.0),
    )
    yield client
    await client.aclose()
