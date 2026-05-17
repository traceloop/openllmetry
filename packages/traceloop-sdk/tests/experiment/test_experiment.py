import pytest
from unittest.mock import Mock, AsyncMock
from traceloop.sdk.experiment.experiment import Experiment
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.evaluator.config import EvaluatorDetails


@pytest.fixture
def experiment():
    """Create an Experiment instance with mocked HTTP client"""
    mock_http_client = Mock(spec=HTTPClient)
    mock_http_client.base_url = "https://api.example.com"
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
    jsonl_data = (
        '{"columns":{"name":{"name":"Name","type":"string"},'
        '"age":{"name":"Age","type":"number"}}}\n'
        '{"name": "John", "age": 30}\n'
        'invalid json line\n'
        '{"name": "Alice", "age": 25}\n'
        '{broken json\n'
        '{"name": "Bob", "age": 35}'
    )

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
    jsonl_data = (
        '{"columns":{"name":{"name":"Name","type":"string"},'
        '"age":{"name":"Age","type":"number"}}}\n'
        '{"name": "John", "age": 30}\n'
        '\n'
        '{"name": "Alice", "age": 25}\n'
        '\n'
    )

    result = experiment._parse_jsonl_to_rows(jsonl_data)

    expected = [{"name": "John", "age": 30}, {"name": "Alice", "age": 25}]
    assert result == expected


def test_parse_jsonl_to_rows_complex_json_objects(experiment):
    """Test parsing JSONL data with complex nested objects"""
    jsonl_data = (
        '{"columns":{"user":{"name":"User","type":"object"},'
        '"active":{"name":"Active","type":"boolean"}}}\n'
        '{"user": {"name": "John", "details": {"age": 30, "location": ["NY", "US"]}}, '
        '"active": true}\n'
        '{"user": {"name": "Alice", "details": {"age": 25, "location": ["CA", "US"]}}, '
        '"active": false}'
    )

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


class TestRunLocallyValidation:
    """Tests for _run_locally to ensure validation failures are handled correctly"""

    @pytest.mark.anyio(backends=["asyncio"])
    async def test_run_locally_breaks_on_validation_failure_with_stop_on_error(self):
        """Test that _run_locally stops processing when validation fails and stop_on_error=True"""
        mock_http_client = Mock(spec=HTTPClient)
        mock_http_client.base_url = "https://api.example.com"
        mock_async_http_client = Mock()

        # Mock the _init_experiment method
        mock_experiment_response = Mock()
        mock_experiment_response.run.id = "run-123"
        mock_experiment_response.experiment.id = "exp-456"

        experiment = Experiment(mock_http_client, mock_async_http_client, "test-exp")
        experiment._init_experiment = Mock(return_value=mock_experiment_response)

        # Create a task that returns output missing required fields
        async def task_missing_fields(row):
            return {"wrong_field": "value"}

        # Define evaluator with required fields
        evaluators = [
            EvaluatorDetails(slug="pii-detector", required_input_fields=["text"]),
        ]

        # Mock dataset response with multiple rows
        experiment._datasets.get_version_jsonl = Mock(
            return_value=(
                '{"columns":{}}\n{"input": "test1"}\n'
                '{"input": "test2"}\n{"input": "test3"}'
            )
        )

        # Run with stop_on_error=True - should stop after first error
        results, errors = await experiment._run_locally(
            task=task_missing_fields,
            evaluators=evaluators,
            dataset_slug="test-dataset",
            dataset_version="v1",
            stop_on_error=True,
        )

        # Should have at least one error and should have broken early
        assert len(errors) >= 1
        assert "Task output missing required fields for evaluators:" in errors[0]
        assert "pii-detector requires:" in errors[0]
        assert "'text'" in errors[0]
        # With stop_on_error=True, it should break and not process all 3 rows
        assert len(results) + len(errors) <= 3

    @pytest.mark.anyio(backends=["asyncio"])
    async def test_run_locally_continues_on_validation_failure_without_stop_on_error(self):
        """Test that _run_locally captures error when validation fails and stop_on_error=False"""
        mock_http_client = Mock(spec=HTTPClient)
        mock_http_client.base_url = "https://api.example.com"
        mock_async_http_client = Mock()

        # Mock the _init_experiment method
        mock_experiment_response = Mock()
        mock_experiment_response.run.id = "run-123"
        mock_experiment_response.experiment.id = "exp-456"

        experiment = Experiment(mock_http_client, mock_async_http_client, "test-exp")
        experiment._init_experiment = Mock(return_value=mock_experiment_response)

        # Create a task that returns output missing required fields
        async def task_missing_fields(row):
            return {"wrong_field": "value"}

        # Define evaluator with required fields
        evaluators = [
            EvaluatorDetails(slug="pii-detector", required_input_fields=["text"]),
        ]

        # Mock dataset response with single row
        experiment._datasets.get_version_jsonl = Mock(
            return_value='{"columns":{}}\n{"input": "test"}'
        )

        # Run with stop_on_error=False - should capture error but not raise
        results, errors = await experiment._run_locally(
            task=task_missing_fields,
            evaluators=evaluators,
            dataset_slug="test-dataset",
            dataset_version="v1",
            stop_on_error=False,
        )

        # Should have errors, no successful results
        assert len(errors) == 1
        assert len(results) == 0
        assert "Task output missing required fields for evaluators:" in errors[0]
        assert "pii-detector requires:" in errors[0]
        assert "'text'" in errors[0]

    @pytest.mark.anyio(backends=["asyncio"])
    async def test_run_locally_succeeds_with_valid_output(self):
        """Test that _run_locally succeeds when task output matches required fields"""
        mock_http_client = Mock(spec=HTTPClient)
        mock_http_client.base_url = "https://api.example.com"
        mock_async_http_client = Mock()

        # Mock the _init_experiment method
        mock_experiment_response = Mock()
        mock_experiment_response.run.id = "run-123"
        mock_experiment_response.experiment.id = "exp-456"

        experiment = Experiment(mock_http_client, mock_async_http_client, "test-exp")
        experiment._init_experiment = Mock(return_value=mock_experiment_response)

        # Mock _create_task
        mock_task_response = Mock()
        mock_task_response.id = "task-789"
        experiment._create_task = Mock(return_value=mock_task_response)

        # Create a task that returns valid output
        async def task_valid_output(row):
            return {"text": "hello world", "score": 0.9}

        # Define evaluator with required fields
        evaluators = [
            EvaluatorDetails(slug="pii-detector", required_input_fields=["text"]),
        ]

        # Mock dataset response
        experiment._datasets.get_version_jsonl = Mock(
            return_value='{"columns":{}}\n{"input": "test"}'
        )

        # Mock evaluator execution
        mock_eval_result = Mock()
        mock_eval_result.result = {"score": 0.95}
        experiment._evaluator.run_experiment_evaluator = AsyncMock(
            return_value=mock_eval_result
        )

        # Run - should succeed
        results, errors = await experiment._run_locally(
            task=task_valid_output,
            evaluators=evaluators,
            dataset_slug="test-dataset",
            dataset_version="v1",
        )

        # Should have successful result, no errors
        assert len(results) == 1
        assert len(errors) == 0
        assert results[0].task_result == {"text": "hello world", "score": 0.9}

    @pytest.mark.anyio(backends=["asyncio"])
    async def test_run_locally_validation_with_multiple_evaluators(self):
        """Test validation with multiple evaluators having different required fields"""
        mock_http_client = Mock(spec=HTTPClient)
        mock_http_client.base_url = "https://api.example.com"
        mock_async_http_client = Mock()

        # Mock the _init_experiment method
        mock_experiment_response = Mock()
        mock_experiment_response.run.id = "run-123"
        mock_experiment_response.experiment.id = "exp-456"

        experiment = Experiment(mock_http_client, mock_async_http_client, "test-exp")
        experiment._init_experiment = Mock(return_value=mock_experiment_response)

        # Create a task that returns partial output
        async def task_partial_output(row):
            return {"text": "hello"}  # Missing 'prompt' and 'response'

        # Define multiple evaluators with different required fields
        evaluators = [
            EvaluatorDetails(slug="pii-detector", required_input_fields=["text"]),
            EvaluatorDetails(
                slug="relevance-checker",
                required_input_fields=["prompt", "response"],
            ),
        ]

        # Mock dataset response
        experiment._datasets.get_version_jsonl = Mock(
            return_value='{"columns":{}}\n{"input": "test"}'
        )

        # Run with stop_on_error=True - should stop after error
        results, errors = await experiment._run_locally(
            task=task_partial_output,
            evaluators=evaluators,
            dataset_slug="test-dataset",
            dataset_version="v1",
            stop_on_error=True,
        )

        # Should have error with validation message
        # Note: 'text' in task output maps to 'response' via synonym mapping,
        # so only 'prompt' is missing
        assert len(errors) >= 1
        error_message = errors[0]
        assert "relevance-checker requires:" in error_message
        assert "'prompt'" in error_message

    @pytest.mark.anyio(backends=["asyncio"])
    async def test_run_locally_no_validation_for_string_evaluators(self):
        """Test that validation is not performed for string evaluators (only EvaluatorDetails)"""
        mock_http_client = Mock(spec=HTTPClient)
        mock_http_client.base_url = "https://api.example.com"
        mock_async_http_client = Mock()

        # Mock the _init_experiment method
        mock_experiment_response = Mock()
        mock_experiment_response.run.id = "run-123"
        mock_experiment_response.experiment.id = "exp-456"

        experiment = Experiment(mock_http_client, mock_async_http_client, "test-exp")
        experiment._init_experiment = Mock(return_value=mock_experiment_response)

        # Mock _create_task
        mock_task_response = Mock()
        mock_task_response.id = "task-789"
        experiment._create_task = Mock(return_value=mock_task_response)

        # Create a task that returns any output
        async def task_any_output(row):
            return {"random_field": "value"}

        # Use string evaluators (no validation)
        evaluators = ["pii-detector", "relevance-checker"]

        # Mock dataset response
        experiment._datasets.get_version_jsonl = Mock(
            return_value='{"columns":{}}\n{"input": "test"}'
        )

        # Mock evaluator execution
        mock_eval_result = Mock()
        mock_eval_result.result = {"score": 0.95}
        experiment._evaluator.run_experiment_evaluator = AsyncMock(
            return_value=mock_eval_result
        )

        # Run - should succeed without validation
        results, errors = await experiment._run_locally(
            task=task_any_output,
            evaluators=evaluators,
            dataset_slug="test-dataset",
            dataset_version="v1",
        )

        # Should succeed because string evaluators don't trigger validation
        assert len(errors) == 0
        assert len(results) == 1
