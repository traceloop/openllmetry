from unittest.mock import Mock, patch
from traceloop.sdk.datasets.model import ColumnType
from traceloop.sdk.datasets.column import Column
from tests.datasets.mock_objects import create_simple_mock_dataset,create_dataset_with_existing_columns
from tests.datasets.mock_response import basic_dataset_response_json
import json

    
@patch('traceloop.sdk.datasets.dataset.Dataset._get_http_client')
def test_add_column_to_empty_dataset(mock_get_http_client):
    """Test adding a new column to an empty dataset"""
    mock_http_client = Mock()
    mock_get_http_client.return_value = mock_http_client
    
    mock_http_client.post.return_value = {"id": "new_column_id"}
    
    dataset, _ = create_simple_mock_dataset()
    
    new_column = dataset.add_column("Test Column", ColumnType.STRING)
    
    mock_http_client.post.assert_called_once_with(
        "projects/default/datasets/test_dataset_id/columns",
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
    
    mock_http_client.post.return_value = {"id": "new_column_id_2"}
    
    dataset, existing_columns = create_dataset_with_existing_columns()
    existing_column_1, existing_column_2 = existing_columns
    
    assert len(dataset.columns) == 2
    
    new_column = dataset.add_column("New Boolean Column", ColumnType.BOOLEAN)
    
    mock_http_client.post.assert_called_once_with(
        "projects/default/datasets/test_dataset_id/columns",
        {"name": "New Boolean Column", "type": ColumnType.BOOLEAN}
    )
    
    assert isinstance(new_column, Column)
    assert new_column.id == "new_column_id_2"
    assert new_column.name == "New Boolean Column"
    assert new_column.type == ColumnType.BOOLEAN
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
    """Test updating a column name using basic_dataset_response_json"""
    mock_http_client = Mock()
    mock_get_http_client.return_value = mock_http_client
    
    dataset_data = json.loads(basic_dataset_response_json)
    
    mock_http_client.put.return_value = {"success": True}
    
    column_id = "cmdwq9a320000coitckjwfpj4"
    column_data = dataset_data["columns"][column_id]
    
    column = Column(
        id=column_id,
        name="old_name",
        type=ColumnType.STRING,
        dataset_id=dataset_data["id"]
    )
    
    new_name = dataset_data["columns"][column_id]["name"]
    column.update(name=new_name, type=ColumnType.NUMBER)
    
    mock_http_client.put.assert_called_once_with(
        f"projects/default/datasets/{dataset_data['id']}/columns/{column_id}",
        {"name": new_name}
    )
    
    assert column.name == new_name
    assert column.id == column_id
    assert column.type == ColumnType.NUMBER
    assert column.dataset_id == dataset_data["id"]