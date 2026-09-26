import pytest


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "api-key", "x-api-key"],
        "filter_query_parameters": ["api_key"],
        "ignore_hosts": ["openaipublic.blob.core.windows.net"],
    }
