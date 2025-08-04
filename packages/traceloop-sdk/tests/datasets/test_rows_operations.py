import json
import pytest
from unittest.mock import patch, MagicMock
from traceloop.sdk.datasets.dataset import Dataset
from traceloop.sdk.datasets.model import ColumnDefinition, ColumnType, CreateDatasetRequest
from .mock_response import create_dataset_response, add_rows_response_json, single_row_response_json 
from .mock_objects import (
    create_mock_dataset_with_columns,
    create_simple_mock_dataset,
    get_test_rows_data
)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_add_rows_to_dataset():
    """Test adding rows to an existing dataset"""
    with patch.object(Dataset, '_get_http_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.post.return_value = json.loads(add_rows_response_json)
        mock_get_client.return_value = mock_client
        
        dataset, _ = create_mock_dataset_with_columns()
                
        test_rows = get_test_rows_data()
        
        result = dataset.add_rows(test_rows)
        
        # Verify the response
        assert result.total == 4
        assert len(result.rows) == 4
        
        # Verify that the dataset now contains the new rows
        assert len(dataset.rows) == 4
        
        # Check all added rows
        expected_rows = [
            ("row_add_1", {"cmdr3ce1s0003hmp0vqons5ey": "Gal", "cmdr3ce1s0004hmp0ies575jr": 8, "cmdr3ce1s0005hmp0bdln01js": True}),
            ("row_add_2", {"cmdr3ce1s0003hmp0vqons5ey": "Nir", "cmdr3ce1s0004hmp0ies575jr": 70, "cmdr3ce1s0005hmp0bdln01js": False}),
            ("row_add_3", {"cmdr3ce1s0003hmp0vqons5ey": "Nina", "cmdr3ce1s0004hmp0ies575jr": 52, "cmdr3ce1s0005hmp0bdln01js": True}),
            ("row_add_4", {"cmdr3ce1s0003hmp0vqons5ey": "Aviv", "cmdr3ce1s0004hmp0ies575jr": 52, "cmdr3ce1s0005hmp0bdln01js": False})
        ]
        
        for row_index, (expected_id, expected_values) in enumerate(expected_rows):
            assert_row_values(dataset, row_index, expected_id, expected_values)
        
        assert mock_client.post.call_count == 1


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_add_single_row():
    """Test adding a single row"""
    with patch.object(Dataset, '_get_http_client') as mock_get_client:
        mock_client = MagicMock()
        
        # Mock response for single row
    
        
        mock_client.post.side_effect = [
            create_dataset_response(),  # create_dataset call
            json.loads(single_row_response_json)  # add_rows call
        ]
        mock_get_client.return_value = mock_client
        
        # Create a simple dataset using mock helper
        dataset, columns_definition = create_simple_mock_dataset()
        
        # Create the dataset
        dataset_response = dataset.create_dataset(CreateDatasetRequest(
            slug="test-dataset",
            name="Test Dataset",
            description="Test dataset",
            columns=columns_definition
        ))
        
        dataset._create_columns(dataset_response.columns)
        
        # Add single row
        test_row = [{"cmdr3ce1s0003hmp0vqons5ey": "single"}]
        result = dataset.add_rows(test_row)
        
        # Verify the response
        assert result.total == 1
        assert len(result.rows) == 1
        assert len(dataset.rows) == 1
        
        # Check the row
        row = dataset.rows[0]
        assert row.id == "single_row_id"
        assert row.row_index == 1
        assert row.values["cmdr3ce1s0003hmp0vqons5ey"] == "single"


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_add_rows_api_failure():
    """Test handling of API failure when adding rows"""
    with patch.object(Dataset, '_get_http_client') as mock_get_client:
        mock_client = MagicMock()
        
        mock_client.post.side_effect = [
            create_dataset_response(),  # create_dataset call
            None  # add_rows call returns None (API failure)
        ]
        mock_get_client.return_value = mock_client
        
        # Create a simple dataset using mock helper
        dataset, columns_definition = create_simple_mock_dataset()
        
        # Create the dataset
        dataset_response = dataset.create_dataset(CreateDatasetRequest(
            slug="test-dataset", 
            name="Test Dataset",
            description="Test dataset",
            columns=columns_definition
        ))
        
        dataset._create_columns(dataset_response.columns)
        
        # Test that add_rows raises exception on API failure
        with pytest.raises(Exception) as exc_info:
            dataset.add_rows([{"cmdr3ce1s0003hmp0vqons5ey": "test"}])
        
        assert "Failed to add row to dataset" in str(exc_info.value)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_add_rows_mixed_types():
    """Test adding rows with mixed data types (string, number, boolean)"""
    with patch.object(Dataset, '_get_http_client') as mock_get_client:
        mock_client = MagicMock()
        
        # Mock response with mixed data types
        mixed_types_response = json.loads(add_rows_response_json)
        
        mock_client.post.return_value = mixed_types_response
        mock_get_client.return_value = mock_client
        
        dataset, columns_definition = create_mock_dataset_with_columns()        
        
        # Add row with mixed types
        test_row = [{
            "cmdr3ce1s0003hmp0vqons5ey": "test_string",
            "cmdr3ce1s0004hmp0ies575jr": 42.5,
            "cmdr3ce1s0005hmp0bdln01js": True
        }]
        result = dataset.add_rows(test_row)
        
        # Verify the response
        assert result.total == 4
        assert len(result.rows) == 4
        assert len(dataset.rows) == 4
        
        # Check the row values and types
        expected_values = {
            "cmdr3ce1s0003hmp0vqons5ey": "Gal",
            "cmdr3ce1s0004hmp0ies575jr": 8,
            "cmdr3ce1s0005hmp0bdln01js": True
        }
        assert_row_values(dataset, 0, "row_add_1", expected_values)
        
        row = dataset.rows[0]
        assert isinstance(row.values["cmdr3ce1s0003hmp0vqons5ey"], str)
        assert isinstance(row.values["cmdr3ce1s0004hmp0ies575jr"], (int, float))
        assert isinstance(row.values["cmdr3ce1s0005hmp0bdln01js"], bool)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_delete_row():
    """Test deleting a row via dataset.rows[index].delete()"""
    with patch.object(Dataset, '_get_http_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.post.return_value = json.loads(add_rows_response_json)
        
        mock_client.delete.return_value = True
        mock_get_client.return_value = mock_client
        
        dataset, _ = create_mock_dataset_with_columns()
        
        test_rows = get_test_rows_data()
        dataset.add_rows(test_rows)
        
        # Verify we have 4 rows initially
        assert len(dataset.rows) == 4
        
        # Get the second row (index 1)
        row_to_delete = dataset.rows[1]
        row_id_to_delete = row_to_delete.id
        
        # Delete the row
        row_to_delete.delete()
        
        # Verify the delete API was called correctly
        mock_client.delete.assert_called_with(
            f"projects/default/datasets/{dataset.slug}/rows/{row_id_to_delete}", 
            {}
        )
        
        # Verify the row was removed from the dataset
        assert len(dataset.rows) == 3
        
        # Verify the specific row was removed
        remaining_row_ids = [row.id for row in dataset.rows]
        assert row_id_to_delete not in remaining_row_ids


def assert_row_values(dataset, row_index, expected_id, expected_values):
    """Helper function to assert row values and reduce code duplication"""
    row = dataset.rows[row_index] if row_index < len(dataset.rows) else None
    assert row is not None
    assert row.id == expected_id
    for column_id, expected_value in expected_values.items():
        assert row.values[column_id] == expected_value
        