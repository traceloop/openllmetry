import json
from unittest.mock import MagicMock, patch
from traceloop.sdk.datasets.dataset import Dataset
from traceloop.sdk.datasets.model import ColumnDefinition, ColumnType, CreateDatasetRequest
from .mock_response import create_dataset_response, add_rows_response_json, create_rows_response_json


def create_mock_dataset_with_columns():
    """Create a mock dataset with standard test columns"""
    columns_definition = [
        ColumnDefinition(name="name", type=ColumnType.STRING),
        ColumnDefinition(name="value", type=ColumnType.NUMBER),
        ColumnDefinition(name="active", type=ColumnType.BOOLEAN)
    ]
    
    dataset = Dataset(
        name="Test Dataset",
        slug="test-dataset",
        description="Test dataset",
        columns_definition=columns_definition
    )
    
    return dataset, columns_definition


def setup_mock_http_client_for_dataset_creation():
    """Setup mock HTTP client for dataset creation operations"""
    mock_client = MagicMock()
    mock_client.post.side_effect = [
        create_dataset_response(),  # create_dataset call
        json.loads(add_rows_response_json)  # add_rows call
    ]
    return mock_client


def create_and_setup_test_dataset():
    """Create a test dataset and set it up with mocked API calls"""
    dataset, columns_definition = create_mock_dataset_with_columns()
    
    # Create the dataset on API
    dataset_response = dataset.create_dataset(CreateDatasetRequest(
        slug="test-dataset",
        name="Test Dataset",
        description="Test dataset",
        columns=columns_definition
    ))
    
    # Create columns from response
    dataset._create_columns(dataset_response.columns)
    
    return dataset, columns_definition


def create_simple_mock_dataset():
    """Create a mock dataset with just one column for simple tests"""
    columns_definition = [
        ColumnDefinition(name="name", type=ColumnType.STRING)
    ]
    
    dataset = Dataset(
        name="Test Dataset",
        slug="test-dataset",
        description="Test dataset",
        columns_definition=columns_definition
    )
    
    return dataset, columns_definition


def get_test_rows_data():
    """Get test row data for testing"""
    return [
        {"cmdr3ce1s0003hmp0vqons5ey": "Gal", "cmdr3ce1s0004hmp0ies575jr": 8, "cmdr3ce1s0005hmp0bdln01js": True},
        {"cmdr3ce1s0003hmp0vqons5ey": "Nir", "cmdr3ce1s0004hmp0ies575jr": 70, "cmdr3ce1s0005hmp0bdln01js": False},
        {"cmdr3ce1s0003hmp0vqons5ey": "Nina", "cmdr3ce1s0004hmp0ies575jr": 52, "cmdr3ce1s0005hmp0bdln01js": True},
        {"cmdr3ce1s0003hmp0vqons5ey": "Aviv", "cmdr3ce1s0004hmp0ies575jr": 52, "cmdr3ce1s0005hmp0bdln01js": True},
    ]


def setup_mock_http_client_for_get_rows():
    """Setup mock HTTP client for getting existing rows"""
    mock_client = MagicMock()
    mock_client.post.side_effect = [
        create_dataset_response(),  # create_dataset call
        json.loads(create_rows_response_json)  # get existing rows
    ]
    return mock_client
