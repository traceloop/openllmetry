import json
from unittest.mock import patch, MagicMock
from traceloop.sdk.datasets.datasets import Datasets
from traceloop.sdk.dataset.dataset import Dataset
from traceloop.sdk.dataset.model import DatasetMetadata
from .mock_response import (get_dataset_by_version_json,
                            get_all_datasets_json,
                            get_dataset_by_slug_json)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_get_dataset_by_slug():
    with patch('traceloop.sdk.client.http.HTTPClient') as mock_http_client_class:
        mock_client = MagicMock()
        mock_client.get.return_value = json.loads(get_dataset_by_slug_json)
        mock_http_client_class.return_value = mock_client

        datasets = Datasets(mock_client)
        dataset = datasets.get_by_slug("product-inventory-2")

        assert isinstance(dataset, Dataset)
        assert dataset.id == "cmdvki9zv003c01vvj7is4p80"
        assert dataset.slug == "product-inventory-2"
        assert dataset.name == "Product Inventory"
        assert dataset.description == "Sample product inventory data"
        assert len(dataset.columns) == 4
        assert len(dataset.rows) == 4

        product_column = next((col for col in dataset.columns if col.name == "product"), None)
        assert product_column is not None
        assert product_column.type == "string"

        price_column = next((col for col in dataset.columns if col.name == "price"), None)
        assert price_column is not None
        assert price_column.type == "number"

        laptop_row = dataset.rows[0]
        assert laptop_row is not None
        assert laptop_row.values["cmdvki9zv003801vv1idaywus"] == "Laptop"
        assert laptop_row.values["cmdvki9zv003901vv5zr5i24b"] == 999.99
        assert laptop_row.values["cmdvki9zv003a01vvvqqlytpr"] is True

        mock_client.get.assert_called_once_with("datasets/product-inventory-2")


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_get_all_datasets():
    with patch('traceloop.sdk.client.http.HTTPClient') as mock_http_client_class:
        mock_client = MagicMock()
        mock_response = json.loads(get_all_datasets_json)
        mock_client.get.return_value = mock_response["datasets"]
        mock_http_client_class.return_value = mock_client

        datasets = Datasets(mock_client)
        datasets_list = datasets.get_all()

        assert isinstance(datasets_list, list)
        assert len(datasets_list) == 6

        # Check first dataset
        first_dataset = datasets_list[0]
        assert isinstance(first_dataset, DatasetMetadata)
        assert first_dataset.id == "cmdwnop4y0004meitkf17oxtn"
        assert first_dataset.slug == "product-inventory-3"
        assert first_dataset.name == "Product Inventory"
        assert first_dataset.description == "Sample product inventory data"

        # Check last dataset
        last_dataset = datasets_list[-1]
        assert last_dataset.id == "cmdvfm9ms001f01vvbe30fbuj"
        assert last_dataset.slug == "employee-data"
        assert last_dataset.name == "Employee Dataset"

        mock_client.get.assert_called_once_with("datasets")


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_get_version_csv():
    with patch('traceloop.sdk.client.http.HTTPClient') as mock_http_client_class:
        mock_client = MagicMock()
        mock_client.get.return_value = get_dataset_by_version_json
        mock_http_client_class.return_value = mock_client

        datasets = Datasets(mock_client)
        csv_data = datasets.get_version_csv(slug="product-inventory", version="v1")

        assert isinstance(csv_data, str)

        mock_client.get.assert_called_once_with("datasets/product-inventory/versions/v1")


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_delete_by_slug():
    with patch('traceloop.sdk.client.http.HTTPClient') as mock_http_client_class:
        mock_client = MagicMock()
        mock_client.delete.return_value = True
        mock_http_client_class.return_value = mock_client

        datasets = Datasets(mock_client)
        datasets.delete_by_slug("test-dataset")

        mock_client.delete.assert_called_once_with("datasets/test-dataset")


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_delete_by_slug_failure():
    with patch('traceloop.sdk.client.http.HTTPClient') as mock_http_client_class:
        mock_client = MagicMock()
        mock_client.delete.return_value = False
        mock_http_client_class.return_value = mock_client

        datasets = Datasets(mock_client)
        
        try:
            datasets.delete_by_slug("test-dataset")
            assert False, "Expected exception was not raised"
        except Exception as e:
            assert "Failed to delete dataset test-dataset" in str(e)

        mock_client.delete.assert_called_once_with("datasets/test-dataset")