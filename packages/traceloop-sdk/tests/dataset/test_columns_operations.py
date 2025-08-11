from unittest.mock import Mock
import pytest
from traceloop.sdk.dataset.model import ColumnType
from traceloop.sdk.dataset.column import Column

from .mock_objects import (
    create_simple_mock_dataset,
    create_dataset_with_existing_columns,
)
from ..fixtures.dataset_responses import (
    BASIC_DATASET_RESPONSE,
    ADD_COLUMN_RESPONSE,
)


def test_add_column_to_empty_dataset():
    """Test adding a new column to an empty dataset"""
    dataset, _ = create_simple_mock_dataset()
    dataset._http.post.return_value = ADD_COLUMN_RESPONSE

    new_column = dataset.add_column("test_column", "Test Column", ColumnType.STRING)

    dataset._http.post.assert_called_once_with(
        f"datasets/{dataset.slug}/columns",
        {"slug": "test_column", "name": "Test Column", "type": ColumnType.STRING},
    )

    assert isinstance(new_column, Column)
    assert new_column.slug == "new-column-id"
    assert new_column.name == "Test Column"
    assert new_column.type == ColumnType.STRING
    assert new_column.dataset_id == "test_dataset_id"

    assert new_column in dataset.columns
    assert len(dataset.columns) == 1


def test_add_column_to_dataset_with_existing_columns():
    """Test adding a new column to a dataset that already has columns"""
    dataset, existing_columns = create_dataset_with_existing_columns()
    dataset._http.post.return_value = ADD_COLUMN_RESPONSE
    existing_column_1, existing_column_2 = existing_columns

    assert len(dataset.columns) == 2

    new_column = dataset.add_column("test_column", "Test Column", ColumnType.STRING)

    dataset._http.post.assert_called_once_with(
        f"datasets/{dataset.slug}/columns",
        {"slug": "test_column", "name": "Test Column", "type": ColumnType.STRING},
    )

    assert isinstance(new_column, Column)
    assert new_column.slug == "new-column-id"
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


def test_update_column():
    """Test updating a column name from dataset.columns[0].update() using BASIC_DATASET_RESPONSE"""
    dataset, _ = create_dataset_with_existing_columns()
    dataset._http.put.return_value = BASIC_DATASET_RESPONSE

    column_to_update = dataset.columns[0]

    new_name = BASIC_DATASET_RESPONSE["columns"]["col-id-1"]["name"]
    new_type = BASIC_DATASET_RESPONSE["columns"]["col-id-1"]["type"]

    dataset.columns[0].update(name=new_name, type=new_type)

    dataset._http.put.assert_called_once_with(
        f"datasets/{dataset.slug}/columns/{column_to_update.slug}",
        {"name": new_name, "type": new_type},
    )

    assert dataset.columns[0].name == new_name
    assert dataset.columns[0].type == new_type
    assert dataset.columns[0].slug == column_to_update.slug
    assert dataset.columns[0].dataset_id == dataset.id


def test_delete_column():
    """Test deleting a column from dataset"""
    dataset, existing_columns = create_dataset_with_existing_columns()
    existing_column_1, existing_column_2 = existing_columns
    
    # Mock API returns status 200 with no body
    dataset._http.delete.return_value = True

    # Add some rows with data for both columns
    from traceloop.sdk.dataset.row import Row

    row1 = Row(
        id="row_id_1",
        values={existing_column_1.slug: "test_value_1", existing_column_2.slug: 42},
        dataset_id="test_dataset_id",
        dataset=dataset,
        http=dataset._http,
    )
    row2 = Row(
        id="row_id_2",
        values={existing_column_1.slug: "test_value_2", existing_column_2.slug: 84},
        dataset_id="test_dataset_id",
        dataset=dataset,
        http=dataset._http,
    )
    dataset.rows = [row1, row2]

    assert len(dataset.columns) == 2
    assert len(dataset.rows) == 2
    assert existing_column_1.slug in row1.values
    assert existing_column_1.slug in row2.values

    existing_column_1.delete()

    # Verify API was called correctly
    dataset._http.delete.assert_called_once_with(
        f"datasets/{dataset.slug}/columns/column_slug_1"
    )

    # Verify column was removed from dataset
    assert len(dataset.columns) == 1
    assert existing_column_1 not in dataset.columns
    assert existing_column_2 in dataset.columns

    # Verify column values were removed from all rows
    assert existing_column_1.slug not in row1.values
    assert existing_column_1.slug not in row2.values

    # Verify other column values remain intact
    assert existing_column_2.slug in row1.values
    assert existing_column_2.slug in row2.values
    assert row1.values[existing_column_2.slug] == 42
    assert row2.values[existing_column_2.slug] == 84
