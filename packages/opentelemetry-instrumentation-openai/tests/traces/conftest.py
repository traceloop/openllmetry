import pytest


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            "authorization",
            "api-key",
            "openai-organization",
            "openai-project",
            "set-cookie",
            "x-request-id",
        ],
        "ignore_hosts": ["openaipublic.blob.core.windows.net"],
    }
