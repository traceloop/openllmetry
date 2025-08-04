import json
from unittest.mock import patch, MagicMock
from traceloop.sdk.datasets.dataset import Dataset
from traceloop.sdk.datasets.model import DatasetMetadata
from .mock_response import get_dataset_by_slug_json, get_all_datasets_json


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_get_dataset_by_slug():
    with patch.object(Dataset, '_get_http_client_static') as mock_get_client:
        mock_client = MagicMock()
        mock_client.get.return_value = json.loads(get_dataset_by_slug_json)
        mock_get_client.return_value = mock_client

        dataset = Dataset.get_by_slug("product-inventory-2")

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

        laptop_row = next((row for row in dataset.rows if row.row_index == 1), None)
        assert laptop_row is not None
        assert laptop_row.values["cmdvki9zv003801vv1idaywus"] == "Laptop"
        assert laptop_row.values["cmdvki9zv003901vv5zr5i24b"] == 999.99
        assert laptop_row.values["cmdvki9zv003a01vvvqqlytpr"] is True

        mock_client.get.assert_called_once_with("projects/default/datasets/product-inventory-2")


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_get_all_datasets():
    with patch.object(Dataset, '_get_http_client_static') as mock_get_client:
        mock_client = MagicMock()
        mock_response = json.loads(get_all_datasets_json)
        mock_client.get.return_value = mock_response["datasets"]
        mock_get_client.return_value = mock_client

        datasets = Dataset.get_all()

        assert isinstance(datasets, list)
        assert len(datasets) == 6

        # Check first dataset
        first_dataset = datasets[0]
        assert isinstance(first_dataset, DatasetMetadata)
        assert first_dataset.id == "cmdwnop4y0004meitkf17oxtn"
        assert first_dataset.slug == "product-inventory-3"
        assert first_dataset.name == "Product Inventory"
        assert first_dataset.description == "Sample product inventory data"

        # Check last dataset
        last_dataset = datasets[-1]
        assert last_dataset.id == "cmdvfm9ms001f01vvbe30fbuj"
        assert last_dataset.slug == "employee-data"
        assert last_dataset.name == "Employee Dataset"

        mock_client.get.assert_called_once_with("projects/default/datasets")
