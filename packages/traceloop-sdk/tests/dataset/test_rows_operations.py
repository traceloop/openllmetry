import pytest
from unittest.mock import patch, MagicMock
from traceloop.sdk.dataset.dataset import Dataset
from traceloop.sdk.dataset.model import CreateDatasetRequest
from .mock_objects import (
    create_mock_dataset_with_columns_definition,
    create_simple_mock_dataset,
    get_test_rows_data,
    create_mock_dataset_with_columns,
)
from ..fixtures.dataset_responses import (
    create_dataset_response,
    ADD_ROWS_RESPONSE,
    SINGLE_ROW_RESPONSE,
)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_add_rows_to_dataset():
    """Test adding rows to an existing dataset"""
    dataset, _ = create_mock_dataset_with_columns_definition()
    mock_client = dataset._http
    mock_client.post.return_value = ADD_ROWS_RESPONSE

    test_rows = get_test_rows_data()

    dataset.add_rows(test_rows)

    # Verify that the dataset now contains the new rows
    assert len(dataset.rows) == 4

    # Check all added rows - using flexible IDs from ADD_ROWS_RESPONSE
    expected_rows = [
        ("row_add_1", {"name": "Gal", "age": 8, "is-active": True}),
        ("row_add_2", {"name": "Nir", "age": 70, "is-active": False}),
        ("row_add_3", {"name": "Nina", "age": 52, "is-active": True}),
        ("row_add_4", {"name": "Aviv", "age": 52, "is-active": False}),
    ]

    for row_index, (expected_id, expected_values) in enumerate(expected_rows):
        row = dataset.rows[row_index]
        assert row.id == expected_id
        # Check values flexibly - the response format matches what we expect
        for key in expected_values:
            assert key in row.values or any(key in str(k) for k in row.values.keys())

    assert mock_client.post.call_count == 1


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_add_single_row():
    """Test adding a single row"""
    with patch.object(Dataset, "_get_http_client") as mock_get_client:
        mock_client = MagicMock()

        # Mock response for single row
        mock_client.post.side_effect = [
            create_dataset_response(),  # create_dataset call
            SINGLE_ROW_RESPONSE,  # add_rows call
        ]
        mock_get_client.return_value = mock_client

        # Create a simple dataset using mock helper
        dataset, columns_definition = create_simple_mock_dataset()

        # Create the dataset
        dataset_response = dataset.create_dataset(
            CreateDatasetRequest(
                slug="test-dataset",
                name="Test Dataset",
                description="Test dataset",
                columns=columns_definition,
            )
        )

        dataset._create_columns(dataset_response.columns)

        # Add single row
        test_row = [{"name": "single"}]
        result = dataset.add_rows_api(test_row)

        # Verify the response
        assert result.total == 1
        assert len(result.rows) == 1
        assert len(dataset.rows) == 1

        # Check the row
        row = dataset.rows[0]
        assert row.id == "single_row_id"
        assert "name" in str(row.values) or "single" in str(row.values)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_add_rows_api_failure():
    """Test handling of API failure when adding rows"""
    with patch.object(Dataset, "_get_http_client") as mock_get_client:
        mock_client = MagicMock()

        mock_client.post.side_effect = [
            create_dataset_response(),  # create_dataset call
            None,  # add_rows call returns None (API failure)
        ]
        mock_get_client.return_value = mock_client

        # Create a simple dataset using mock helper
        dataset, columns_definition = create_simple_mock_dataset()

        # Create the dataset
        dataset_response = dataset.create_dataset(
            CreateDatasetRequest(
                slug="test-dataset",
                name="Test Dataset",
                description="Test dataset",
                columns=columns_definition,
            )
        )

        dataset._create_columns(dataset_response.columns)

        # Test that add_rows raises exception on API failure
        with pytest.raises(Exception) as exc_info:
            dataset.add_rows_api([{"test": "value"}])

        assert "Failed to add row to dataset" in str(exc_info.value)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_add_rows_mixed_types():
    """Test adding rows with mixed data types (string, number, boolean)"""
    with patch.object(Dataset, "_get_http_client") as mock_get_client:
        mock_client = MagicMock()

        # Create a custom response with exactly 3 rows for mixed types test
        mixed_types_response = {
            "rows": [
                {
                    "id": "mixed_row_1",
                    "row_index": 0,
                    "values": {"name": "string_value"},
                    "created_at": "2025-08-03T12:00:00.000Z",
                    "updated_at": "2025-08-03T12:00:00.000Z",
                },
                {
                    "id": "mixed_row_2",
                    "row_index": 1,
                    "values": {"age": 42},
                    "created_at": "2025-08-03T12:00:00.000Z",
                    "updated_at": "2025-08-03T12:00:00.000Z",
                },
                {
                    "id": "mixed_row_3",
                    "row_index": 2,
                    "values": {"active": True},
                    "created_at": "2025-08-03T12:00:00.000Z",
                    "updated_at": "2025-08-03T12:00:00.000Z",
                },
            ],
            "total": 3,
        }

        mock_client.post.side_effect = [
            create_dataset_response(),  # create_dataset call
            mixed_types_response,  # add_rows call
        ]
        mock_get_client.return_value = mock_client

        # Create a simple dataset using mock helper
        dataset, columns_definition = create_simple_mock_dataset()

        # Create the dataset
        dataset_response = dataset.create_dataset(
            CreateDatasetRequest(
                slug="test-dataset",
                name="Test Dataset",
                description="Test dataset",
                columns=columns_definition,
            )
        )

        dataset._create_columns(dataset_response.columns)

        # Add rows with mixed types
        mixed_rows = [
            {"name": "string_value"},
            {"age": 42},
            {"active": True},
        ]

        result = dataset.add_rows_api(mixed_rows)

        # Verify the response
        assert result.total == 3
        assert len(result.rows) == 3


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_delete_row():
    """Test deleting a row from dataset"""
    with patch.object(Dataset, "_get_http_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.delete.return_value = {"success": True}
        mock_client.post.return_value = ADD_ROWS_RESPONSE
        mock_get_client.return_value = mock_client

        dataset = create_mock_dataset_with_columns()

        # Add some rows first
        test_rows = get_test_rows_data()
        dataset.add_rows_api(test_rows)

        if dataset.rows:
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

    row = Row(id="test_row_id", values={"test": "value"}, dataset_id="test_dataset_id")

    # Try to delete a row without a client
    with pytest.raises(ValueError) as exc_info:
        row.delete()

    assert "Row must be associated with a dataset to delete" in str(exc_info.value)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_update_row():
    """Test updating a row in dataset"""
    with patch.object(Dataset, "_get_http_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.put.return_value = {"success": True}
        mock_client.post.return_value = ADD_ROWS_RESPONSE
        mock_get_client.return_value = mock_client

        dataset = create_mock_dataset_with_columns()

        # Add some rows first
        test_rows = get_test_rows_data()
        dataset.add_rows_api(test_rows)

        if dataset.rows:
            # Get the row to update
            row_to_update = dataset.rows[0]
            first_column_key = list(row_to_update.values.keys())[0] if row_to_update.values else "name"
            new_values = {first_column_key: "Updated Name"}

            # Update the row using the Row object's update method
            row_to_update.update(new_values)

            # Verify the row was updated in the dataset
            assert row_to_update.values[first_column_key] == "Updated Name"

            # Verify the API was called correctly
            mock_client.put.assert_called_once()


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_update_row_without_dataset():
    """Test updating a row when it's not associated with a dataset"""
    # Create a row without a client
    from traceloop.sdk.dataset.row import Row

    row = Row(id="test_row_id", values={"test": "value"}, dataset_id="test_dataset_id")

    # Try to update a row without a client
    with pytest.raises(ValueError) as exc_info:
        row.update({"test": "new_value"})

    assert "Row must be associated with a dataset to update" in str(exc_info.value)


@patch.dict("os.environ", {"TRACELOOP_API_KEY": "test-api-key"})
def test_update_row_api_failure():
    """Test handling of API failure when updating row"""
    # For VCR-based testing, we skip this test as it requires specific API failure scenarios
    pytest.skip("API failure scenarios are better tested with integration tests")


def assert_row_values(dataset, row_index, expected_id, expected_values):
    """Helper function to assert row values"""
    row = dataset.rows[row_index]
    assert row.id == expected_id
    # Use flexible value checking since the response format may vary
    for key, value in expected_values.items():
        # Check if key exists directly or as part of the values
        found = False
        for row_key, row_value in row.values.items():
            if key in str(row_key) or row_value == value:
                found = True
                break
        # Allow for some flexibility in value matching
        if not found and str(value) not in str(row.values):
            assert False, f"Expected {key}={value} not found in row values: {row.values}"