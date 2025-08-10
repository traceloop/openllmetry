import json
from unittest.mock import patch, MagicMock
from traceloop.sdk.dataset.dataset import Dataset
from traceloop.sdk.dataset.model import DatasetMetadata
from .mock_response import (publish_dataset_response_json,
                            get_dataset_by_version_json,
                            get_all_datasets_json,
                            get_dataset_by_slug_json)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_get_dataset_by_version():
    with patch('traceloop.sdk.client.http.HTTPClient') as mock_http_client_class:
        mock_client = MagicMock()
        mock_client.get.return_value = get_dataset_by_version_json
        mock_http_client_class.return_value = mock_client

        dataset = Dataset(http=mock_client)
        dataset.slug = "product-inventory-2"
        csv_data = dataset.get_version_csv(slug="product-inventory", version="v1")

        assert isinstance(csv_data, str)

        mock_client.get.assert_called_once_with("datasets/product-inventory/versions/v1")


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_publish_dataset():
    with patch('traceloop.sdk.client.http.HTTPClient') as mock_http_client_class:
        mock_client = MagicMock()
        mock_client.post.return_value = json.loads(publish_dataset_response_json)
        mock_http_client_class.return_value = mock_client

        dataset = Dataset(http=mock_client)
        dataset.slug = "test-dataset"
        version = dataset.publish()

        assert version == "v1"
        mock_client.post.assert_called_once_with("datasets/test-dataset/publish", {})
