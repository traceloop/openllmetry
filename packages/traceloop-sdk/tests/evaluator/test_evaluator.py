import pytest
from traceloop.sdk.evaluator.evaluator import validate_and_normalize_task_output
from traceloop.sdk.evaluator.config import EvaluatorDetails


class TestValidateTaskOutput:
    """Tests for validate_task_output function"""

    def test_validate_task_output_with_no_evaluators(self):
        """Test that validation passes when no evaluators are provided"""
        task_output = {"text": "hello"}
        evaluators = []

        # Should not raise any exception
        validate_and_normalize_task_output(task_output, evaluators)

    def test_validate_task_output_with_evaluators_no_required_fields(self):
        """Test that validation passes when evaluators have no required fields"""
        task_output = {"text": "hello", "score": 0.9}
        evaluators = [
            EvaluatorDetails(slug="evaluator1"),
            EvaluatorDetails(slug="evaluator2", config={"threshold": 0.5}),
        ]

        # Should not raise any exception
        validate_and_normalize_task_output(task_output, evaluators)

    def test_validate_task_output_with_valid_output(self):
        """Test that validation passes when all required fields are present"""
        task_output = {"text": "hello", "prompt": "say hello", "response": "world"}
        evaluators = [
            EvaluatorDetails(slug="pii-detector", required_input_fields=["text"]),
            EvaluatorDetails(
                slug="relevance-checker",
                required_input_fields=["prompt", "response"],
            ),
        ]

        # Should not raise any exception
        validate_and_normalize_task_output(task_output, evaluators)

    def test_validate_task_output_missing_single_field(self):
        """Test that validation fails when a single required field is missing"""
        task_output = {"prompt": "say hello"}
        evaluators = [
            EvaluatorDetails(slug="pii-detector", required_input_fields=["text"]),
        ]

        with pytest.raises(ValueError) as exc_info:
            validate_and_normalize_task_output(task_output, evaluators)

        error_message = str(exc_info.value)
        assert "Task output missing required fields for evaluators:" in error_message
        assert "pii-detector requires:" in error_message
        assert "'text'" in error_message
        assert "Task output contains: ['prompt']" in error_message
        assert (
            "Hint: Update your task function to return a dictionary "
            "with the required fields."
        ) in error_message

    def test_validate_task_output_missing_multiple_fields_single_evaluator(self):
        """Test that validation fails when multiple fields are missing for one evaluator"""
        task_output = {"score": 0.9}
        evaluators = [
            EvaluatorDetails(
                slug="relevance-checker",
                required_input_fields=["prompt", "response", "context"],
            ),
        ]

        with pytest.raises(ValueError) as exc_info:
            validate_and_normalize_task_output(task_output, evaluators)

        error_message = str(exc_info.value)
        assert "relevance-checker requires:" in error_message
        assert "'context'" in error_message
        assert "'prompt'" in error_message
        assert "'response'" in error_message

    def test_validate_task_output_missing_fields_multiple_evaluators(self):
        """Test that validation fails when fields are missing for multiple evaluators"""
        task_output = {"score": 0.9}
        evaluators = [
            EvaluatorDetails(slug="pii-detector", required_input_fields=["text"]),
            EvaluatorDetails(
                slug="relevance-checker",
                required_input_fields=["prompt", "response"],
            ),
            EvaluatorDetails(slug="tone-analyzer", required_input_fields=["text"]),
        ]

        with pytest.raises(ValueError) as exc_info:
            validate_and_normalize_task_output(task_output, evaluators)

        error_message = str(exc_info.value)
        assert "pii-detector requires:" in error_message
        assert "relevance-checker requires:" in error_message
        assert "tone-analyzer requires:" in error_message
        assert "'text'" in error_message
        assert "'prompt'" in error_message
        assert "'response'" in error_message

    def test_validate_task_output_partial_match(self):
        """Test validation when some evaluators pass and some fail"""
        task_output = {"text": "hello world"}
        evaluators = [
            EvaluatorDetails(slug="pii-detector", required_input_fields=["text"]),
            EvaluatorDetails(
                slug="relevance-checker",
                required_input_fields=["prompt", "response"],
            ),
        ]

        with pytest.raises(ValueError) as exc_info:
            validate_and_normalize_task_output(task_output, evaluators)

        error_message = str(exc_info.value)
        # Should only mention the failing evaluator
        assert "relevance-checker requires:" in error_message
        assert "pii-detector requires:" not in error_message

    def test_validate_task_output_empty_task_output(self):
        """Test validation with empty task output"""
        task_output = {}
        evaluators = [
            EvaluatorDetails(slug="pii-detector", required_input_fields=["text"]),
        ]

        with pytest.raises(ValueError) as exc_info:
            validate_and_normalize_task_output(task_output, evaluators)

        error_message = str(exc_info.value)
        assert "Task output contains: []" in error_message

    def test_validate_task_output_with_extra_fields(self):
        """Test that validation passes when task output has extra fields"""
        task_output = {
            "text": "hello",
            "prompt": "say hello",
            "response": "world",
            "extra_field": "value",
            "another_field": 123,
        }
        evaluators = [
            EvaluatorDetails(slug="pii-detector", required_input_fields=["text"]),
        ]

        # Should not raise any exception - extra fields are allowed
        validate_and_normalize_task_output(task_output, evaluators)

    def test_validate_task_output_case_sensitive_field_names(self):
        """Test that field name matching is case-sensitive"""
        task_output = {"Text": "hello"}
        evaluators = [
            EvaluatorDetails(slug="pii-detector", required_input_fields=["text"]),
        ]

        with pytest.raises(ValueError) as exc_info:
            validate_and_normalize_task_output(task_output, evaluators)

        error_message = str(exc_info.value)
        assert "pii-detector requires:" in error_message
        assert "'text'" in error_message
        assert "Task output contains: ['Text']" in error_message

    def test_validate_task_output_with_evaluator_config(self):
        """Test validation with evaluators that have config"""
        task_output = {"text": "hello world"}
        evaluators = [
            EvaluatorDetails(
                slug="pii-detector",
                version="v2",
                config={"probability_threshold": 0.8, "mode": "strict"},
                required_input_fields=["text"],
            ),
        ]

        # Should not raise any exception - config shouldn't affect validation
        validate_and_normalize_task_output(task_output, evaluators)

    def test_validate_task_output_mixed_evaluators(self):
        """Test validation with a mix of evaluators with and without required fields"""
        task_output = {"text": "hello", "score": 0.9}
        evaluators = [
            EvaluatorDetails(slug="evaluator-no-requirements"),
            EvaluatorDetails(slug="pii-detector", required_input_fields=["text"]),
            EvaluatorDetails(slug="another-no-requirements", config={"key": "value"}),
            EvaluatorDetails(
                slug="relevance-checker",
                required_input_fields=["prompt"],
            ),
        ]

        with pytest.raises(ValueError) as exc_info:
            validate_and_normalize_task_output(task_output, evaluators)

        error_message = str(exc_info.value)
        # Should only mention failing evaluator
        assert "relevance-checker requires:" in error_message
        assert "'prompt'" in error_message
        assert "evaluator-no-requirements" not in error_message
        assert "pii-detector" not in error_message or "pii-detector requires:" not in error_message

    def test_validate_task_output_duplicate_required_fields(self):
        """Test validation when multiple evaluators require the same field"""
        task_output = {"score": 0.9}
        evaluators = [
            EvaluatorDetails(slug="pii-detector", required_input_fields=["text"]),
            EvaluatorDetails(slug="tone-analyzer", required_input_fields=["text"]),
            EvaluatorDetails(
                slug="sentiment-analyzer",
                required_input_fields=["text", "language"],
            ),
        ]

        with pytest.raises(ValueError) as exc_info:
            validate_and_normalize_task_output(task_output, evaluators)

        error_message = str(exc_info.value)
        assert "pii-detector requires:" in error_message
        assert "tone-analyzer requires:" in error_message
        assert "sentiment-analyzer requires:" in error_message
