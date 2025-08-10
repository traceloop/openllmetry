import json
from unittest.mock import MagicMock
from traceloop.sdk.dataset.dataset import Dataset
from traceloop.sdk.dataset.model import ColumnDefinition, ColumnType
from traceloop.sdk.dataset.column import Column
from .mock_response import (
    create_dataset_response,
    add_rows_response_json,
    create_rows_response_json,
)


def create_mock_dataset_with_columns_definition():
    """Create a mock dataset with standard test columns"""
    columns_definition = [
        ColumnDefinition(name="name", type=ColumnType.STRING),
        ColumnDefinition(name="value", type=ColumnType.NUMBER),
        ColumnDefinition(name="active", type=ColumnType.BOOLEAN),
    ]

    mock_http_client = MagicMock()
    dataset = Dataset(http=mock_http_client)
    dataset.id = "mock-dataset-id"
    dataset.name = "Test Dataset"
    dataset.slug = "test-dataset"
    dataset.description = "Test dataset"

    return dataset, columns_definition


def create_mock_dataset_with_columns():
    """Create a mock dataset with standard test columns"""
    dataset, columns_definition = create_mock_dataset_with_columns_definition()

    raw_columns = {
        "column_id_" + str(i): column for i, column in enumerate(columns_definition)
    }
    dataset._create_columns(raw_columns)

    return dataset


def setup_mock_http_client_for_dataset_creation():
    """Setup mock HTTP client for dataset creation operations"""
    mock_client = MagicMock()
    mock_client.post.side_effect = [
        create_dataset_response(),  # create_dataset call
        json.loads(add_rows_response_json),  # add_rows call
    ]
    return mock_client


def create_simple_mock_dataset():
    """Create a mock dataset with just one column for simple tests"""
    columns_definition = [ColumnDefinition(name="name", type=ColumnType.STRING)]

    mock_http_client = MagicMock()
    dataset = Dataset(http=mock_http_client)
    dataset.id = "test_dataset_id"
    dataset.name = "Test Dataset"
    dataset.slug = "test-dataset"
    dataset.description = "Test dataset"

    return dataset, columns_definition


def get_test_rows_data():
    """Get test row data for testing"""
    return [
        {
            "cmdr3ce1s0003hmp0vqons5ey": "Gal",
            "cmdr3ce1s0004hmp0ies575jr": 8,
            "cmdr3ce1s0005hmp0bdln01js": True,
        },
        {
            "cmdr3ce1s0003hmp0vqons5ey": "Nir",
            "cmdr3ce1s0004hmp0ies575jr": 70,
            "cmdr3ce1s0005hmp0bdln01js": False,
        },
        {
            "cmdr3ce1s0003hmp0vqons5ey": "Nina",
            "cmdr3ce1s0004hmp0ies575jr": 52,
            "cmdr3ce1s0005hmp0bdln01js": True,
        },
        {
            "cmdr3ce1s0003hmp0vqons5ey": "Aviv",
            "cmdr3ce1s0004hmp0ies575jr": 52,
            "cmdr3ce1s0005hmp0bdln01js": False,
        },
    ]


def setup_mock_http_client_for_get_rows():
    """Setup mock HTTP client for getting existing rows"""
    mock_client = MagicMock()
    mock_client.post.side_effect = [
        create_dataset_response(),  # create_dataset call
        json.loads(create_rows_response_json),  # get existing rows
    ]
    return mock_client


def create_dataset_with_existing_columns():
    """Create a dataset with existing columns for testing column operations"""
    mock_http_client = MagicMock()
    dataset = Dataset(http=mock_http_client)
    dataset.id = "test_dataset_id"
    dataset.slug = "test-dataset"
    dataset.name = "Test Dataset"

    # Add existing columns to the dataset
    existing_column_1 = Column(
        http=mock_http_client,
        dataset=dataset,
        id="column_id_1",
        name="Column 1",
        type=ColumnType.STRING,
        dataset_id="test_dataset_id",
    )
    existing_column_2 = Column(
        http=mock_http_client,
        dataset=dataset,
        id="column_id_2",
        name="Column 2",
        type=ColumnType.NUMBER,
        dataset_id="test_dataset_id",
    )

    dataset.columns.extend([existing_column_1, existing_column_2])

    return dataset, [existing_column_1, existing_column_2]
