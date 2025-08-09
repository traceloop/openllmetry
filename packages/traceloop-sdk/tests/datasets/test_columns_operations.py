from unittest.mock import Mock, patch
from traceloop.sdk.datasets.model import ColumnType
from traceloop.sdk.datasets.column import Column

from tests.datasets.mock_objects import create_simple_mock_dataset, create_dataset_with_existing_columns
from tests.datasets.mock_response import basic_dataset_response_json, add_column_response_json
import json


@patch('traceloop.sdk.datasets.dataset.Dataset._get_http_client')
def test_add_column_to_empty_dataset(mock_get_http_client):
    """Test adding a new column to an empty dataset"""
    mock_http_client = Mock()
    mock_get_http_client.return_value = mock_http_client

    mock_http_client.post.return_value = add_column_response_json

    dataset, _ = create_simple_mock_dataset()

    new_column = dataset.add_column("Test Column", ColumnType.STRING)

    mock_http_client.post.assert_called_once_with(
        f"projects/default/datasets/{dataset.slug}/columns",
        {"name": "Test Column", "type": ColumnType.STRING}
    )

    assert isinstance(new_column, Column)
    assert new_column.id == "new_column_id"
    assert new_column.name == "Test Column"
    assert new_column.type == ColumnType.STRING
    assert new_column.dataset_id == "test_dataset_id"

    assert new_column in dataset.columns
    assert len(dataset.columns) == 1


@patch('traceloop.sdk.datasets.dataset.Dataset._get_http_client')
def test_add_column_to_dataset_with_existing_columns(mock_get_http_client):
    """Test adding a new column to a dataset that already has columns"""
    mock_http_client = Mock()
    mock_get_http_client.return_value = mock_http_client

    mock_http_client.post.return_value = add_column_response_json

    dataset, existing_columns = create_dataset_with_existing_columns()
    existing_column_1, existing_column_2 = existing_columns

    assert len(dataset.columns) == 2

    new_column = dataset.add_column("Test Column", ColumnType.STRING)

    mock_http_client.post.assert_called_once_with(
        f"projects/default/datasets/{dataset.slug}/columns",
        {"name": "Test Column", "type": ColumnType.STRING}
    )

    assert isinstance(new_column, Column)
    assert new_column.id == "new_column_id"
    assert new_column.name == "Test Column"
    assert new_column.type == ColumnType.STRING
    assert new_column.dataset_id == "test_dataset_id"

    assert new_column in dataset.columns
    assert len(dataset.columns) == 3

    assert existing_column_1 in dataset.columns
    assert existing_column_2 in dataset.columns

    assert dataset.columns[0] == existing_column_1
    assert dataset.columns[1] == existing_column_2
    assert dataset.columns[2] == new_column


@patch('traceloop.sdk.datasets.dataset.Dataset._get_http_client')
def test_update_column(mock_get_http_client):
    """Test updating a column name from dataset.columns[0].update() using basic_dataset_response_json"""
    mock_http_client = Mock()
    mock_get_http_client.return_value = mock_http_client

    mock_http_client.put.return_value = json.loads(basic_dataset_response_json)

    dataset, _ = create_dataset_with_existing_columns()

    column_to_update = dataset.columns[0]

    response_data = json.loads(basic_dataset_response_json)
    new_name = response_data["columns"]["column_id_1"]["name"]
    new_type = response_data["columns"]["column_id_1"]["type"]

    dataset.columns[0].update(name=new_name, type=new_type)

    mock_http_client.put.assert_called_once_with(
        f"projects/default/datasets/{dataset.slug}/columns/{column_to_update.id}",
        {"name": new_name, "type": new_type}
    )

    assert dataset.columns[0].name == new_name
    assert dataset.columns[0].type == new_type
    assert dataset.columns[0].id == column_to_update.id
    assert dataset.columns[0].dataset_id == dataset.id


@patch('traceloop.sdk.datasets.dataset.Dataset._get_http_client')
def test_delete_column(mock_get_http_client):
    """Test deleting a column from dataset"""
    mock_http_client = Mock()
    mock_get_http_client.return_value = mock_http_client

    # Mock API returns status 200 with no body
    mock_http_client.delete.return_value = True

    dataset, existing_columns = create_dataset_with_existing_columns()
    existing_column_1, existing_column_2 = existing_columns

    # Add some rows with data for both columns
    from traceloop.sdk.datasets.row import Row
    row1 = Row(
        id="row_id_1",
        values={existing_column_1.id: "test_value_1", existing_column_2.id: 42},
        dataset_id="test_dataset_id",
        _client=dataset
    )
    row2 = Row(
        id="row_id_2",
        row_index=1,
        values={existing_column_1.id: "test_value_2", existing_column_2.id: 84},
        dataset_id="test_dataset_id",
        _client=dataset
    )
    dataset.rows = [row1, row2]

    assert len(dataset.columns) == 2
    assert len(dataset.rows) == 2
    assert existing_column_1.id in row1.values
    assert existing_column_1.id in row2.values

    existing_column_1.delete()

    # Verify API was called correctly
    mock_http_client.delete.assert_called_once_with(
        f"projects/default/datasets/{dataset.slug}/columns/column_id_1"
    )

    # Verify column was removed from dataset
    assert len(dataset.columns) == 1
    assert existing_column_1 not in dataset.columns
    assert existing_column_2 in dataset.columns

    # Verify column values were removed from all rows
    assert existing_column_1.id not in row1.values
    assert existing_column_1.id not in row2.values

    # Verify other column values remain intact
    assert existing_column_2.id in row1.values
    assert existing_column_2.id in row2.values
    assert row1.values[existing_column_2.id] == 42
    assert row2.values[existing_column_2.id] == 84
