def create_dataset_response(price_type="string", in_stock_type="string"):
    """Generate dataset response with configurable column types"""
    return {
        "id": "cmdvei5dd000g01vvyftz2zv1",
        "orgId": "c108269c-cf1e-4ac6-a7e4-5a456cc9fdb7",
        "projectId": "cm9v2g95l0011z613sv851kwd",
        "slug": "daatset-12",
        "name": "Dataset",
        "description": "Dataset Description",
        "columns": {
            "cmdvei5dd000d01vv2yvmp7vt": {
                "name": "Name",
                "type": "string"
            },
            "cmdvei5dd000e01vvz0eb5kz8": {
                "name": "Price",
                "type": price_type
            },
            "cmdvei5dd000f01vv7aazk674": {
                "name": "In Stock",
                "type": in_stock_type
            }
        },
        "lastVersion": None,
        "createdAt": "2025-08-03T08:09:53.329521779Z",
        "updatedAt": "2025-08-03T08:09:53.329522049Z"
    }

create_rows_response_json = """
{
    "rows": [
        {
            "id": "row_1_id",
            "rowIndex": 1,
            "values": {
                "cmdvei5dd000d01vv2yvmp7vt": "Laptop",
                "cmdvei5dd000e01vvz0eb5kz8": 999.99,
                "cmdvei5dd000f01vv7aazk674": true
            },
            "created_at": "2025-08-03T08:10:00.000Z",
            "updated_at": "2025-08-03T08:10:00.000Z"
        },
        {
            "id": "row_2_id",
            "rowIndex": 2,
            "values": {
                "cmdvei5dd000d01vv2yvmp7vt": "Mouse",
                "cmdvei5dd000e01vvz0eb5kz8": 29.99,
                "cmdvei5dd000f01vv7aazk674": false
            },
            "created_at": "2025-08-03T08:10:00.000Z",
            "updated_at": "2025-08-03T08:10:00.000Z"
        }
    ],
    "total": 2
}
"""

import json
import pytest
import tempfile
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from traceloop.sdk.datasets.dataset import Dataset


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_create_dataset_from_csv():
    # Create temporary CSV file
    csv_content = """Name,Price,In Stock
    Laptop,999.99,true
    Mouse,29.99,false"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_path = f.name
    
    try:
        with patch.object(Dataset, '_get_http_client') as mock_get_client:
            mock_client = MagicMock()
            
            # Mock dataset creation response
            mock_client.post.side_effect = [
                create_dataset_response(),  # CSV: all string types
                json.loads(create_rows_response_json)      # add_rows call
            ]
            mock_get_client.return_value = mock_client
            
            dataset = Dataset.from_csv(
                file_path=csv_path,
                slug="test-csv-dataset",
                name="Test CSV Dataset",
                description="Dataset created from CSV"
            )
            
            assert isinstance(dataset, Dataset)
            assert dataset.id == "cmdvei5dd000g01vvyftz2zv1"
            assert dataset.slug == "daatset-12"
            assert dataset.name == "Dataset"
            assert dataset.description == "Dataset Description"
            assert len(dataset.columns) == 3
            assert len(dataset.rows) == 2
            
            name_column = next((col for col in dataset.columns if col.name == "Name"), None)
            assert name_column is not None
            assert name_column.type == "string"
            
            price_column = next((col for col in dataset.columns if col.name == "Price"), None)
            assert price_column is not None
            assert price_column.type == "string"  # CSV columns are always string initially
            
            laptop_row = next((row for row in dataset.rows if row.row_index == 1), None)
            assert laptop_row is not None
            assert laptop_row.values["cmdvei5dd000d01vv2yvmp7vt"] == "Laptop"
            
            assert mock_client.post.call_count == 2
            
    finally:
        os.unlink(csv_path)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_create_dataset_from_dataframe():
    # Create test dataframe
    df = pd.DataFrame({
        'Name': ['Laptop', 'Mouse'],
        'Price': [999.99, 29.99],
        'In Stock': [True, False]
    })
    
    with patch.object(Dataset, '_get_http_client') as mock_get_client:
        mock_client = MagicMock()
        
        # Mock dataset creation response
        mock_client.post.side_effect = [
            create_dataset_response(price_type="number", in_stock_type="boolean"),  # DataFrame: inferred types
            json.loads(create_rows_response_json)      # add_rows call
        ]
        mock_get_client.return_value = mock_client
        
        dataset = Dataset.from_dataframe(
            df=df,
            slug="test-df-dataset",
            name="Test DataFrame Dataset",
            description="Dataset created from DataFrame"
        )
        
        assert isinstance(dataset, Dataset)
        assert dataset.id == "cmdvei5dd000g01vvyftz2zv1"
        assert dataset.slug == "daatset-12"
        assert dataset.name == "Dataset"
        assert dataset.description == "Dataset Description"
        assert len(dataset.columns) == 3
        assert len(dataset.rows) == 2
        
        name_column = next((col for col in dataset.columns if col.name == "Name"), None)
        assert name_column is not None
        assert name_column.type == "string"
        
        price_column = next((col for col in dataset.columns if col.name == "Price"), None)
        assert price_column is not None
        assert price_column.type == "number"
        
        stock_column = next((col for col in dataset.columns if col.name == "In Stock"), None)
        assert stock_column is not None
        assert stock_column.type == "boolean"
        
        laptop_row = next((row for row in dataset.rows if row.row_index == 1), None)
        assert laptop_row is not None
        assert laptop_row.values["cmdvei5dd000d01vv2yvmp7vt"] == "Laptop"
        
        assert mock_client.post.call_count == 2


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_create_dataset_from_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        Dataset.from_csv(
            file_path="/nonexistent/file.csv",
            slug="test-dataset",
            name="Test Dataset"
        )

