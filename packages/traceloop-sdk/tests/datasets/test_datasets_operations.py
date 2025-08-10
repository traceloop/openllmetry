import json
import pytest
from traceloop.sdk.dataset.dataset import Dataset
from traceloop.sdk.dataset.model import DatasetMetadata
from .mock_response import (get_dataset_by_version_json,
                            get_all_datasets_json,
                            get_dataset_by_slug_json)
from .test_constants import TestConstants


def test_get_dataset_by_slug(datasets, mock_http):
    mock_http.get.return_value = json.loads(get_dataset_by_slug_json)

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
    assert product_column.type == TestConstants.STRING_TYPE

    price_column = next((col for col in dataset.columns if col.name == "price"), None)
    assert price_column is not None
    assert price_column.type == TestConstants.NUMBER_TYPE

    laptop_row = dataset.rows[0]
    assert laptop_row is not None
    assert laptop_row.values["cmdvki9zv003801vv1idaywus"] == "Laptop"
    assert laptop_row.values["cmdvki9zv003901vv5zr5i24b"] == 999.99
    assert laptop_row.values["cmdvki9zv003a01vvvqqlytpr"] is True

    mock_http.get.assert_called_once_with("datasets/product-inventory-2")


def test_get_all_datasets(datasets, mock_http):
    mock_response = json.loads(get_all_datasets_json)
    mock_http.get.return_value = mock_response["datasets"]

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

    mock_http.get.assert_called_once_with("datasets")


def test_get_version_csv(datasets, mock_http):
    mock_http.get.return_value = get_dataset_by_version_json

    csv_data = datasets.get_version_csv(slug="product-inventory", version="v1")

    assert isinstance(csv_data, str)

    mock_http.get.assert_called_once_with("datasets/product-inventory/versions/v1")


def test_delete_by_slug(datasets, mock_http):
    mock_http.delete.return_value = True

    datasets.delete_by_slug("test-dataset")

    mock_http.delete.assert_called_once_with("datasets/test-dataset")


def test_delete_by_slug_failure(datasets, mock_http):
    mock_http.delete.return_value = False
    
    with pytest.raises(Exception) as exc_info:
        datasets.delete_by_slug("test-dataset")
    
    assert "Failed to delete dataset test-dataset" in str(exc_info.value)
    mock_http.delete.assert_called_once_with("datasets/test-dataset")


def test_get_all_datasets_failure(datasets, mock_http):
    mock_http.get.return_value = None
    
    with pytest.raises(Exception) as exc_info:
        datasets.get_all()
    
    assert "Failed to get datasets" in str(exc_info.value)
    mock_http.get.assert_called_once_with("datasets")


def test_get_dataset_by_slug_failure(datasets, mock_http):
    mock_http.get.return_value = None
    
    with pytest.raises(Exception) as exc_info:
        datasets.get_by_slug("non-existent-dataset")
    
    assert "Failed to get dataset non-existent-dataset" in str(exc_info.value)
    mock_http.get.assert_called_once_with("datasets/non-existent-dataset")


def test_get_version_csv_failure(datasets, mock_http):
    mock_http.get.return_value = None
    
    with pytest.raises(Exception) as exc_info:
        datasets.get_version_csv("non-existent-dataset", "v1")
    
    assert "Failed to get dataset non-existent-dataset by version v1" in str(exc_info.value)
    mock_http.get.assert_called_once_with("datasets/non-existent-dataset/versions/v1")