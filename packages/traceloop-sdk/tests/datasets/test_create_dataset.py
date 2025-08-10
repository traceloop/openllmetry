import json
import pytest
import tempfile
import os
import pandas as pd
from traceloop.sdk.dataset.dataset import Dataset
from .mock_response import create_dataset_response, create_rows_response_json
from .test_constants import TestConstants


def test_create_dataset_from_csv(datasets, mock_http):
    # Create temporary CSV file
    csv_content = """Name,Price,In Stock
    Laptop,999.99,true
    Mouse,29.99,false"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_path = f.name

    try:
        # Mock dataset creation response
        mock_http.post.side_effect = [
            create_dataset_response(),
            json.loads(create_rows_response_json)
        ]

        dataset = datasets.from_csv(
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
        assert name_column.type == TestConstants.STRING_TYPE

        price_column = next((col for col in dataset.columns if col.name == "Price"), None)
        assert price_column is not None
        assert price_column.type == TestConstants.STRING_TYPE

        laptop_row = dataset.rows[0]
        assert laptop_row is not None
        assert laptop_row.values["cmdvei5dd000d01vv2yvmp7vt"] == "Laptop"

        assert mock_http.post.call_count == 2

    finally:
        os.unlink(csv_path)


def test_create_dataset_from_dataframe(datasets, mock_http):
    # Create test dataframe
    df = pd.DataFrame({
        'Name': ['Laptop', 'Mouse'],
        'Price': [999.99, 29.99],
        'In Stock': [True, False]
    })

    # Mock dataset creation response
    mock_http.post.side_effect = [
        create_dataset_response(price_type=TestConstants.NUMBER_TYPE, in_stock_type=TestConstants.BOOLEAN_TYPE),
        json.loads(create_rows_response_json)
    ]

    dataset = datasets.from_dataframe(
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
    assert name_column.type == TestConstants.STRING_TYPE

    price_column = next((col for col in dataset.columns if col.name == "Price"), None)
    assert price_column is not None
    assert price_column.type == TestConstants.NUMBER_TYPE

    stock_column = next((col for col in dataset.columns if col.name == "In Stock"), None)
    assert stock_column is not None
    assert stock_column.type == TestConstants.BOOLEAN_TYPE

    laptop_row = dataset.rows[0]
    assert laptop_row is not None
    assert laptop_row.values["cmdvei5dd000d01vv2yvmp7vt"] == "Laptop"

    assert mock_http.post.call_count == 2


def test_create_dataset_from_csv_file_not_found(datasets):
    with pytest.raises(FileNotFoundError):
        datasets.from_csv(
            file_path=TestConstants.NON_EXISTENT_FILE_PATH,
            slug=TestConstants.DATASET_SLUG,
            name=TestConstants.DATASET_NAME
        )


def test_create_dataset_from_csv_create_failure(datasets, mock_http):
    # Create temporary CSV file
    csv_content = """Name,Price
    Laptop,999.99"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_path = f.name

    try:
        # Mock HTTP post to return None to simulate failure
        mock_http.post.return_value = None
        
        with pytest.raises(Exception) as exc_info:
            datasets.from_csv(
                file_path=csv_path,
                slug=TestConstants.DATASET_SLUG,
                name=TestConstants.DATASET_NAME
            )
        
        assert "Failed to create dataset" in str(exc_info.value)

    finally:
        os.unlink(csv_path)


def test_create_dataset_from_dataframe_create_failure(datasets, mock_http):
    # Create test dataframe
    df = pd.DataFrame({
        'Name': ['Laptop'],
        'Price': [999.99]
    })

    # Mock HTTP post to return None to simulate failure
    mock_http.post.return_value = None
    
    with pytest.raises(Exception) as exc_info:
        datasets.from_dataframe(
            df=df,
            slug=TestConstants.DATASET_SLUG,
            name=TestConstants.DATASET_NAME
        )
    
    assert "Failed to create dataset" in str(exc_info.value)
