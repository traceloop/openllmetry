import json
import requests
import pytest
from unittest.mock import patch, MagicMock
from traceloop.sdk.dataset.dataset import Dataset
from traceloop.sdk.dataset.model import CreateDatasetRequest
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

        result = dataset.add_rows_api(test_rows)

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
        result = dataset.add_rows_api(test_row)

        # Verify the response
        assert result.total == 1
        assert len(result.rows) == 1
        assert len(dataset.rows) == 1

        # Check the row
        row = dataset.rows[0]
        assert row.id == "single_row_id"
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
            dataset.add_rows_api([{"cmdr3ce1s0003hmp0vqons5ey": "test"}])

        assert "Failed to add row to dataset" in str(exc_info.value)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_add_rows_mixed_types():
    """Test adding rows with mixed data types (string, number, boolean)"""
    with patch.object(Dataset, '_get_http_client') as mock_get_client:
        mock_client = MagicMock()

        # Create a custom response with exactly 3 rows for mixed types test
        mixed_types_response = {
            "rows": [
                {
                    "id": "mixed_row_1",
                    "rowIndex": 0,
                    "values": {"cmdr3ce1s0003hmp0vqons5ey": "string_value"},
                    "created_at": "2025-08-03T12:00:00.000Z",
                    "updated_at": "2025-08-03T12:00:00.000Z"
                },
                {
                    "id": "mixed_row_2",
                    "rowIndex": 1,
                    "values": {"cmdr3ce1s0004hmp0ies575jr": 42},
                    "created_at": "2025-08-03T12:00:00.000Z",
                    "updated_at": "2025-08-03T12:00:00.000Z"
                },
                {
                    "id": "mixed_row_3",
                    "rowIndex": 2,
                    "values": {"cmdr3ce1s0005hmp0bdln01js": True},
                    "created_at": "2025-08-03T12:00:00.000Z",
                    "updated_at": "2025-08-03T12:00:00.000Z"
                }
            ],
            "total": 3
        }

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

        result = dataset.add_rows_api(mixed_rows)

        # Verify the response
        assert result.total == 3
        assert len(result.rows) == 3


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_delete_row():
    """Test deleting a row from dataset"""
    with patch.object(Dataset, '_get_http_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.delete.return_value = {"success": True}
        mock_client.post.return_value = json.loads(add_rows_response_json)
        mock_get_client.return_value = mock_client

        dataset = create_mock_dataset_with_columns()

        # Add some rows first
        test_rows = get_test_rows_data()
        dataset.add_rows_api(test_rows)

        # Get the row to delete
        row_to_delete = dataset.rows[0]
        row_id = row_to_delete.id

        # Delete the row using the Row object's delete method
        row_to_delete.delete()

        # Verify the row was removed from the dataset
        remaining_row_ids = [row.id for row in dataset.rows]
        assert row_id not in remaining_row_ids

        # Verify the API was called correctly
        mock_client.delete.assert_called_once()


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_delete_row_without_client():
    """Test deleting a row when it's not associated with a dataset"""
    # Create a row without a client
    from traceloop.sdk.dataset.row import Row

    row = Row(
        id="test_row_id",
        values={"test": "value"},
        dataset_id="test_dataset_id"
    )

    # Try to delete a row without a client
    with pytest.raises(ValueError) as exc_info:
        row.delete()

    assert "Row must be associated with a dataset to delete" in str(exc_info.value)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_update_row():
    """Test updating a row in dataset"""
    with patch.object(Dataset, '_get_http_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.put.return_value = {"success": True}
        mock_client.post.return_value = json.loads(add_rows_response_json)
        mock_get_client.return_value = mock_client

        dataset = create_mock_dataset_with_columns()

        # Add some rows first
        test_rows = get_test_rows_data()
        dataset.add_rows_api(test_rows)

        # Get the row to update
        row_to_update = dataset.rows[0]
        new_values = {"cmdr3ce1s0003hmp0vqons5ey": "Updated Name"}

        # Update the row using the Row object's update method
        row_to_update.update(new_values)

        # Verify the row was updated in the dataset
        assert row_to_update.values["cmdr3ce1s0003hmp0vqons5ey"] == "Updated Name"

        # Verify the API was called correctly
        mock_client.put.assert_called_once()


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_update_row_without_dataset():
    """Test updating a row when it's not associated with a dataset"""
    # Create a row without a client
    from traceloop.sdk.dataset.row import Row

    row = Row(
        id="test_row_id",
        values={"test": "value"},
        dataset_id="test_dataset_id"
    )

    # Try to update a row without a client
    with pytest.raises(ValueError) as exc_info:
        row.update({"test": "new_value"})

    assert "Row must be associated with a dataset to update" in str(exc_info.value)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_update_row_api_failure():
    """Test handling of API failure when updating row"""
    with patch.object(Dataset, '_get_http_client') as mock_get_client:
        mock_client = MagicMock()
        mock_client.put.side_effect = requests.exceptions.RequestException("Test exception")
        mock_client.post.return_value = json.loads(add_rows_response_json)
        mock_get_client.return_value = mock_client

        dataset = create_mock_dataset_with_columns()

        # Add some rows first
        test_rows = get_test_rows_data()
        dataset.add_rows_api(test_rows)

        # Get the row to update
        row_to_update = dataset.rows[0]
        new_values = {"cmdr3ce1s0003hmp0vqons5ey": "Updated Name"}

        # Try to update a row
        with pytest.raises(requests.exceptions.RequestException):
            row_to_update.update(new_values)


def assert_row_values(dataset, row_index, expected_id, expected_values):
    """Helper function to assert row values"""
    row = dataset.rows[row_index]
    assert row.id == expected_id
    for key, value in expected_values.items():
        assert row.values[key] == value
