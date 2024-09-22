"""Unit tests configuration module."""

import pytest
import os
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI

pytest_plugins = []


@pytest.fixture(autouse=True)
def environment():
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "test_api_key"
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        os.environ["AZURE_OPENAI_API_KEY"] = "test_azure_api_key"
    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://traceloop-stg.openai.azure.com/"


@pytest.fixture
def openai_client():
    return OpenAI()


@pytest.fixture
def azure_openai_client():
    return AzureOpenAI(
        api_version="2024-02-01",
    )


@pytest.fixture
def async_azure_openai_client():
    return AsyncAzureOpenAI(
        api_version="2024-02-01",
    )


@pytest.fixture
def async_openai_client():
    return AsyncOpenAI()


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization", "api-key"]}
