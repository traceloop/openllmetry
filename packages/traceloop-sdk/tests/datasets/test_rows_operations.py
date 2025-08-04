import json
import pytest
from unittest.mock import patch, MagicMock
from traceloop.sdk.datasets.dataset import Dataset
from traceloop.sdk.datasets.model import CreateDatasetRequest
from .mock_response import (
    create_dataset_response,
    add_rows_response_json,
    single_row_response_json
)
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
            ("row_add_1", {
                "cmdr3ce1s0003hmp0vqons5ey": "Gal",
                "cmdr3ce1s0004hmp0ies575jr": 8,
                "cmdr3ce1s0005hmp0bdln01js": True
            }),
            ("row_add_2", {
                "cmdr3ce1s0003hmp0vqons5ey": "Nir",
                "cmdr3ce1s0004hmp0ies575jr": 70,
                "cmdr3ce1s0005hmp0bdln01js": False
            }),
            ("row_add_3", {
                "cmdr3ce1s0003hmp0vqons5ey": "Nina",
                "cmdr3ce1s0004hmp0ies575jr": 52,
                "cmdr3ce1s0005hmp0bdln01js": True
            }),
            ("row_add_4", {
                "cmdr3ce1s0003hmp0vqons5ey": "Aviv",
                "cmdr3ce1s0004hmp0ies575jr": 52,
                "cmdr3ce1s0005hmp0bdln01js": False
            })
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

        mock_client.post.side_effect = [
            create_dataset_response(),  # create_dataset call
            mixed_types_response  # add_rows call
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

        # Add rows with mixed types
        mixed_rows = [
            {"cmdr3ce1s0003hmp0vqons5ey": "string_value"},
            {"cmdr3ce1s0004hmp0ies575jr": 42},
            {"cmdr3ce1s0005hmp0bdln01js": True}
        ]

        result = dataset.add_rows(mixed_rows)

        # Verify the response
        assert result.total == 3
        assert len(result.rows) == 3


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_delete_row():
    """Test deleting a row from dataset"""
    with patch.object(Dataset, '_get_http_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.delete.return_value = {"success": True}
        mock_get_client.return_value = mock_client

        dataset, _ = create_mock_dataset_with_columns()

        # Add some rows first
        test_rows = get_test_rows_data()
        dataset.add_rows(test_rows)

        # Delete a row
        row_id = "row_add_1"
        result = dataset.delete_row(row_id)

        # Verify the response
        assert result["success"] is True

        # Verify the row was removed from the dataset
        remaining_row_ids = [row.id for row in dataset.rows]
        assert row_id not in remaining_row_ids

        # Verify the API was called correctly
        mock_client.delete.assert_called_once()


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_update_row():
    """Test updating a row in dataset"""
    with patch.object(Dataset, '_get_http_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.put.return_value = {"success": True}
        mock_get_client.return_value = mock_client

        dataset, _ = create_mock_dataset_with_columns()

        # Add some rows first
        test_rows = get_test_rows_data()
        dataset.add_rows(test_rows)

        # Update a row
        row_id = "row_add_1"
        new_values = {"cmdr3ce1s0003hmp0vqons5ey": "Updated Name"}

        result = dataset.update_row(row_id, new_values)

        # Verify the response
        assert result["success"] is True

        # Verify the row was updated in the dataset
        updated_row = next((row for row in dataset.rows if row.id == row_id), None)
        assert updated_row is not None
        assert updated_row.values["cmdr3ce1s0003hmp0vqons5ey"] == "Updated Name"

        # Verify the API was called correctly
        mock_client.put.assert_called_once()


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_update_row_without_client():
    """Test updating a row when HTTP client is not available"""
    dataset, _ = create_mock_dataset_with_columns()

    # Add some rows first
    test_rows = get_test_rows_data()
    dataset.add_rows(test_rows)

    # Try to update a row without HTTP client
    row_id = "row_add_1"
    new_values = {"cmdr3ce1s0003hmp0vqons5ey": "Updated Name"}

    with pytest.raises(Exception) as exc_info:
        dataset.update_row(row_id, new_values)

    assert "HTTP client not available" in str(exc_info.value)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_update_row_api_failure():
    """Test handling of API failure when updating row"""
    with patch.object(Dataset, '_get_http_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.put.return_value = None  # API failure
        mock_get_client.return_value = mock_client

        dataset, _ = create_mock_dataset_with_columns()

        # Add some rows first
        test_rows = get_test_rows_data()
        dataset.add_rows(test_rows)

        # Try to update a row
        row_id = "row_add_1"
        new_values = {"cmdr3ce1s0003hmp0vqons5ey": "Updated Name"}

        with pytest.raises(Exception) as exc_info:
            dataset.update_row(row_id, new_values)

        assert "Failed to update row" in str(exc_info.value)


def assert_row_values(dataset, row_index, expected_id, expected_values):
    """Helper function to assert row values"""
    row = dataset.rows[row_index]
    assert row.id == expected_id
    for key, value in expected_values.items():
        assert row.values[key] == value
