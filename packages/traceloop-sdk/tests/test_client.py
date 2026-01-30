import pytest
from traceloop.sdk.client import Client
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.annotation.user_feedback import UserFeedback
from traceloop.sdk.datasets.datasets import Datasets
from traceloop.sdk.experiment import Experiment


def test_client_initialization():
    """Test basic client initialization"""
    client = Client(api_key="test-key", app_name="test-app")

    assert client.app_name == "test-app"
    assert client.api_key == "test-key"
    assert client.api_endpoint == "https://api.traceloop.com"
    assert isinstance(client._http, HTTPClient)


def test_client_custom_endpoint():
    """Test client initialization with custom endpoint"""
    client = Client(api_key="test-key", app_name="test-app", api_endpoint="https://custom.endpoint.com")

    assert client.api_endpoint == "https://custom.endpoint.com"
    assert client._http.base_url == "https://custom.endpoint.com"


def test_client_default_app_name():
    """Test client initialization with default app_name"""
    client = Client(api_key="test-key")

    # Default app_name should be sys.argv[0]
    import sys

    assert client.app_name == sys.argv[0]


@pytest.mark.parametrize("api_key", [None, "", " "])
def test_client_requires_api_key(api_key):
    """Test that client requires a valid API key"""
    with pytest.raises(ValueError, match="API key is required"):
        Client(api_key=api_key)


def test_user_feedback_initialization():
    """Test user_feedback is properly initialized"""
    client = Client(api_key="test-key", app_name="test-app")

    assert isinstance(client.user_feedback, UserFeedback)
    assert client.user_feedback._http == client._http
    assert client.user_feedback._app_name == client.app_name


def test_client_lazy_loads_datasets():
    """Test client.datasets is only initialized lazy."""
    client = Client(api_key="test-key", app_name="test-app")
    assert client._datasets is None  # Initial state is None

    datasets = client.datasets

    assert isinstance(datasets, Datasets)
    assert client._datasets is not None  # Then it's loaded
    assert client.datasets is datasets  # And always returns same instance


def test_datasets_deprecation_warnings():
    """Test client.datasets emits proper deprecation warnings."""
    client = Client(api_key="test-key", app_name="test-app")
    with pytest.deprecated_call():
        client.datasets
    with pytest.deprecated_call():
        client.datasets = Datasets(client._http)
    with pytest.deprecated_call():
        del client.datasets

def test_client_lazy_loads_experiment():
    """Test cilent.experiment is only initialized lazy."""
    client = Client(api_key="test-key", app_name="test-app")
    assert client._experiment is None  # Initial state is None

    experiment = client.experiment

    assert isinstance(experiment, Experiment)
    assert client._experiment is not None  # Then it's loaded
    assert client.experiment is experiment  # And always returns same instance

