import pytest


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "api-key"],
        "ignore_hosts": ["openaipublic.blob.core.windows.net"],
    }
