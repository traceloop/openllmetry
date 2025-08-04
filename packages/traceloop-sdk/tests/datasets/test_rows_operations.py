import json
import pytest
from unittest.mock import patch, MagicMock
from traceloop.sdk.datasets.dataset import Dataset
from traceloop.sdk.datasets.model import ColumnDefinition, ColumnType, CreateDatasetRequest
from .mock_response import create_dataset_response, add_rows_response_json, single_row_response_json 
from .mock_objects import (
    create_mock_dataset_with_columns_definition,
    create_simple_mock_dataset,
    get_test_rows_data,
    create_mock_dataset_with_columns
)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_add_rows_to_dataset():
    """Test adding rows to an existing dataset"""
    with patch.object(Dataset, '_get_http_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.post.return_value = json.loads(add_rows_response_json)
        mock_get_client.return_value = mock_client
        
        dataset, _ = create_mock_dataset_with_columns_definition()
                
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
        
        dataset, columns_definition = create_mock_dataset_with_columns_definition()        
        
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
        
        dataset, _ = create_mock_dataset_with_columns_definition()
        
        test_rows = get_test_rows_data()
        dataset.add_rows(test_rows)
        
        assert len(dataset.rows) == 4
        
        row_to_delete = dataset.rows[1]
        row_id_to_delete = row_to_delete.id
        
        row_to_delete.delete()
        
        mock_client.delete.assert_called_with(
            f"projects/default/datasets/{dataset.slug}/rows/{row_id_to_delete}", 
            {}
        )
        
        assert len(dataset.rows) == 3
        
        remaining_row_ids = [row.id for row in dataset.rows]
        assert row_id_to_delete not in remaining_row_ids


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_update_row():
    """Test updating a row's values"""
    with patch.object(Dataset, '_get_http_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.post.return_value = json.loads(add_rows_response_json)
        
        # Mock the update API response (Go service returns 200 with no body)
        mock_client.put.return_value = {}
        mock_get_client.return_value = mock_client
        
        dataset = create_mock_dataset_with_columns()
        
        test_rows = get_test_rows_data()
        dataset.add_rows(test_rows)
        
        assert len(dataset.rows) == 4
        
        row_to_update = dataset.rows[0]
        
        new_name = "Updated Name"
        new_values = {
            dataset.columns[0].id: new_name,
            dataset.columns[1].id: 99
        }
        
        row_to_update.update(new_values)
        
        # Verify API call was made correctly
        mock_client.put.assert_called_with(
            f"projects/default/datasets/{dataset.slug}/rows/{row_to_update.id}", 
            {"values": new_values}
        )
        
        assert dataset.rows[0].values[dataset.columns[0].id] == new_name
        assert dataset.rows[0].values[dataset.columns[1].id] == 99


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_update_row_without_client():
    """Test updating a row that is not associated with a dataset"""
    from traceloop.sdk.datasets.row import Row
    
    # Create a row without a client
    row = Row(
        id="test_row", 
        row_index=1, 
        values={"test": "value"}, 
        dataset_id="test_dataset"
    )
    
    # Should raise ValueError when trying to update
    with pytest.raises(ValueError) as exc_info:
        row.update({"test": "new_value"})
    
    assert "Row must be associated with a dataset to update" in str(exc_info.value)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_update_row_api_failure():
    """Test handling of API failure when updating a row"""
    with patch.object(Dataset, '_get_http_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.post.return_value = json.loads(add_rows_response_json)
        
        # Mock API failure (returns None)
        mock_client.put.return_value = None
        mock_get_client.return_value = mock_client
        
        dataset, _ = create_mock_dataset_with_columns_definition()
        
        test_rows = get_test_rows_data()
        dataset.add_rows(test_rows)
        
        row_to_update = dataset.rows[0]
        
        # Should raise exception on API failure
        with pytest.raises(Exception) as exc_info:
            row_to_update.update({"cmdr3ce1s0003hmp0vqons5ey": "Updated Name"})
        
        assert f"Failed to update cells in dataset {dataset.slug}" in str(exc_info.value)


def assert_row_values(dataset, row_index, expected_id, expected_values):
    """Helper function to assert row values and reduce code duplication"""
    row = dataset.rows[row_index] if row_index < len(dataset.rows) else None
    assert row is not None
    assert row.id == expected_id
    for column_id, expected_value in expected_values.items():
        assert row.values[column_id] == expected_value
        