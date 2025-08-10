import json
import pytest
import tempfile
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from traceloop.sdk.dataset.dataset import Dataset
from .mock_response import create_dataset_response, create_rows_response_json
from .test_constants import TestConstants


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
                slug=TestConstants.DATASET_SLUG,
                name=TestConstants.DATASET_NAME,
                description=TestConstants.DATASET_DESCRIPTION
            )

            assert isinstance(dataset, Dataset)
            assert dataset.id == "cmdvei5dd000g01vvyftz2zv1"
            assert dataset.slug == TestConstants.DATASET_SLUG
            assert dataset.name == TestConstants.DATASET_NAME
            assert dataset.description == TestConstants.DATASET_DESCRIPTION
            assert len(dataset.columns) == 3
            assert len(dataset.rows) == 2

            name_column = next((col for col in dataset.columns if col.name == "Name"), None)
            assert name_column is not None
            assert name_column.type == "string"

            price_column = next((col for col in dataset.columns if col.name == "Price"), None)
            assert price_column is not None
            assert price_column.type == "string"  # CSV columns are always string initially

            laptop_row = dataset.rows[0]
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
            slug=TestConstants.DATASET_SLUG,
            name=TestConstants.DATASET_NAME,
            description=TestConstants.DATASET_DESCRIPTION
        )

        assert isinstance(dataset, Dataset)
        assert dataset.id == "cmdvei5dd000g01vvyftz2zv1"
        assert dataset.slug == TestConstants.DATASET_SLUG
        assert dataset.name == TestConstants.DATASET_NAME
        assert dataset.description == TestConstants.DATASET_DESCRIPTION
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

        laptop_row = dataset.rows[0]
        assert laptop_row is not None
        assert laptop_row.values["cmdvei5dd000d01vv2yvmp7vt"] == "Laptop"

        assert mock_client.post.call_count == 2


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_create_dataset_from_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        Dataset.from_csv(
            file_path=TestConstants.NON_EXISTENT_FILE_PATH,
            slug=TestConstants.DATASET_SLUG,
            name=TestConstants.DATASET_NAME
        )
