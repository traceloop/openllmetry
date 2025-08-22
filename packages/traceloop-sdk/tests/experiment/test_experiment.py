import pytest
from unittest.mock import Mock
from traceloop.sdk.experiment.experiment import Experiment
from traceloop.sdk.client.http import HTTPClient


@pytest.fixture
def experiment():
    """Create an Experiment instance with mocked HTTP client"""
    mock_http_client = Mock(spec=HTTPClient)
    mock_async_http_client = Mock()
    experiment_slug = "test-experiment"
    return Experiment(mock_http_client, mock_async_http_client, experiment_slug)


def test_parse_jsonl_to_rows_valid_data(experiment):
    """Test parsing valid JSONL data"""
    jsonl_data = """{"columns":{"name":{"name":"Name","type":"string"},
    "age":{"name":"Age","type":"number"},"city":{"name":"City","type":"string"}}}
{"name": "John", "age": 30, "city": "New York"}
{"name": "Alice", "age": 25, "city": "San Francisco"}
{"name": "Bob", "age": 35, "city": "Chicago"}"""

    result = experiment._parse_jsonl_to_rows(jsonl_data)

    expected = [
        {"name": "John", "age": 30, "city": "New York"},
        {"name": "Alice", "age": 25, "city": "San Francisco"},
        {"name": "Bob", "age": 35, "city": "Chicago"},
    ]
    assert result == expected


def test_parse_jsonl_to_rows_with_invalid_json_lines(experiment):
    """Test parsing JSONL data with some invalid JSON lines"""
    jsonl_data = """{"columns":{"name":{"name":"Name","type":"string"},"age":{"name":"Age","type":"number"}}}
{"name": "John", "age": 30}
invalid json line
{"name": "Alice", "age": 25}
{broken json
{"name": "Bob", "age": 35}"""

    result = experiment._parse_jsonl_to_rows(jsonl_data)

    expected = [
        {"name": "John", "age": 30},
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 35},
    ]
    assert result == expected


def test_parse_jsonl_to_rows_empty_input(experiment):
    """Test parsing empty JSONL data"""
    jsonl_data = ""

    result = experiment._parse_jsonl_to_rows(jsonl_data)

    assert result == []


def test_parse_jsonl_to_rows_only_header(experiment):
    """Test parsing JSONL data with only column header"""
    jsonl_data = (
        '{"columns":{"user-description":{"name":"User Description","type":"string"}}}'
    )

    result = experiment._parse_jsonl_to_rows(jsonl_data)

    assert result == []


def test_parse_jsonl_to_rows_with_empty_lines(experiment):
    """Test parsing JSONL data with empty lines"""
    jsonl_data = """{"columns":{"name":{"name":"Name","type":"string"},"age":{"name":"Age","type":"number"}}}
{"name": "John", "age": 30}

{"name": "Alice", "age": 25}

"""

    result = experiment._parse_jsonl_to_rows(jsonl_data)

    expected = [{"name": "John", "age": 30}, {"name": "Alice", "age": 25}]
    assert result == expected


def test_parse_jsonl_to_rows_complex_json_objects(experiment):
    """Test parsing JSONL data with complex nested objects"""
    jsonl_data = """{"columns":{"user":{"name":"User","type":"object"},"active":{"name":"Active","type":"boolean"}}}
{"user": {"name": "John", "details": {"age": 30, "location": ["NY", "US"]}}, "active": true}
{"user": {"name": "Alice", "details": {"age": 25, "location": ["CA", "US"]}}, "active": false}"""

    result = experiment._parse_jsonl_to_rows(jsonl_data)

    expected = [
        {
            "user": {"name": "John", "details": {"age": 30, "location": ["NY", "US"]}},
            "active": True,
        },
        {
            "user": {"name": "Alice", "details": {"age": 25, "location": ["CA", "US"]}},
            "active": False,
        },
    ]
    assert result == expected
