get_dataset_by_slug_json = """
{
    "id": "cmdvki9zv003c01vvj7is4p80",
    "slug": "product-inventory-2",
    "name": "Product Inventory",
    "description": "Sample product inventory data",
    "columns": {
        "cmdvki9zv003801vv1idaywus": {
            "name": "product",
            "type": "string"
        },
        "cmdvki9zv003901vv5zr5i24b": {
            "name": "price",
            "type": "number"
        },
        "cmdvki9zv003a01vvvqqlytpr": {
            "name": "in_stock",
            "type": "boolean"
        },
        "cmdvki9zv003b01vvmk3d22km": {
            "name": "category",
            "type": "string"
        }
    },
    "created_at": "2025-08-03T10:57:57.019Z",
    "updated_at": "2025-08-03T10:57:57.019Z",
    "rows": [
        {
            "id": "cmdvkieye003d01vv1zlmkjrg",
            "rowIndex": 1,
            "values": {
                "cmdvki9zv003801vv1idaywus": "Laptop",
                "cmdvki9zv003901vv5zr5i24b": 999.99,
                "cmdvki9zv003a01vvvqqlytpr": true,
                "cmdvki9zv003b01vvmk3d22km": "Electronics"
            },
            "created_at": "2025-08-03T10:58:03.451Z",
            "updated_at": "2025-08-03T10:58:03.451Z"
        },
        {
            "id": "cmdvkieye003e01vvs4onq0sq",
            "rowIndex": 2,
            "values": {
                "cmdvki9zv003801vv1idaywus": "Mouse",
                "cmdvki9zv003901vv5zr5i24b": 29.99,
                "cmdvki9zv003a01vvvqqlytpr": true,
                "cmdvki9zv003b01vvmk3d22km": "Accessories"
            },
            "created_at": "2025-08-03T10:58:03.451Z",
            "updated_at": "2025-08-03T10:58:03.451Z"
        },
        {
            "id": "cmdvkieye003f01vvkc6jxgad",
            "rowIndex": 3,
            "values": {
                "cmdvki9zv003801vv1idaywus": "Keyboard",
                "cmdvki9zv003901vv5zr5i24b": 79.99,
                "cmdvki9zv003a01vvvqqlytpr": false,
                "cmdvki9zv003b01vvmk3d22km": "Accessories"
            },
            "created_at": "2025-08-03T10:58:03.451Z",
            "updated_at": "2025-08-03T10:58:03.451Z"
        },
        {
            "id": "cmdvkieye003g01vvcuze8p1z",
            "rowIndex": 4,
            "values": {
                "cmdvki9zv003801vv1idaywus": "Monitor",
                "cmdvki9zv003901vv5zr5i24b": 299.99,
                "cmdvki9zv003a01vvvqqlytpr": true,
                "cmdvki9zv003b01vvmk3d22km": "Electronics"
            },
            "created_at": "2025-08-03T10:58:03.451Z",
            "updated_at": "2025-08-03T10:58:03.451Z"
        }
    ]
}
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from traceloop.sdk.datasets.dataset import Dataset


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
        
        laptop_row = next((row for row in dataset.rows if row.index == 1), None)
        assert laptop_row is not None
        assert laptop_row.values["cmdvki9zv003801vv1idaywus"] == "Laptop"
        assert laptop_row.values["cmdvki9zv003901vv5zr5i24b"] == 999.99
        assert laptop_row.values["cmdvki9zv003a01vvvqqlytpr"] is True
        
        mock_client.get.assert_called_once_with("projects/default/datasets/product-inventory-2")