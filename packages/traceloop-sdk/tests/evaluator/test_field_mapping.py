import pytest
from traceloop.sdk.evaluator.field_mapping import (
    get_synonyms,
    normalize_task_output,
    get_field_suggestions,
    format_field_help,
)


class TestGetSynonyms:
    """Tests for get_synonyms function"""

    def test_get_synonyms_for_text(self):
        """Test getting synonyms for 'text' field"""
        synonyms = get_synonyms("text")
        assert "text" in synonyms
        assert "completion" in synonyms
        assert "answer" in synonyms
        assert "response" in synonyms
        assert len(synonyms) == 4

    def test_get_synonyms_for_question(self):
        """Test getting synonyms for 'question' field"""
        synonyms = get_synonyms("question")
        assert "question" in synonyms
        assert "prompt" in synonyms
        assert "instructions" in synonyms
        assert "query" in synonyms
        assert len(synonyms) == 4

    def test_get_synonyms_for_reference(self):
        """Test getting synonyms for 'reference' field"""
        synonyms = get_synonyms("reference")
        assert "reference" in synonyms
        assert "ground_truth" in synonyms
        assert "context" in synonyms
        assert len(synonyms) == 3

    def test_get_synonyms_for_trajectory_prompts(self):
        """Test getting synonyms for 'trajectory_prompts' field"""
        synonyms = get_synonyms("trajectory_prompts")
        assert "trajectory_prompts" in synonyms
        assert "prompts" in synonyms
        assert len(synonyms) == 2

    def test_get_synonyms_for_trajectory_completions(self):
        """Test getting synonyms for 'trajectory_completions' field"""
        synonyms = get_synonyms("trajectory_completions")
        assert "trajectory_completions" in synonyms
        assert "completions" in synonyms
        assert len(synonyms) == 2

    def test_get_synonyms_for_non_synonym_field(self):
        """Test getting synonyms for a field without synonyms"""
        synonyms = get_synonyms("tool_input")
        assert synonyms == {"tool_input"}
        assert len(synonyms) == 1

    def test_get_synonyms_for_unknown_field(self):
        """Test getting synonyms for an unknown field"""
        synonyms = get_synonyms("unknown_field")
        assert synonyms == {"unknown_field"}
        assert len(synonyms) == 1

    def test_get_synonyms_symmetry(self):
        """Test that synonym relationships are symmetric"""
        # All synonyms in a group should map to the same set
        text_synonyms = get_synonyms("text")
        completion_synonyms = get_synonyms("completion")
        answer_synonyms = get_synonyms("answer")
        response_synonyms = get_synonyms("response")

        assert text_synonyms == completion_synonyms
        assert text_synonyms == answer_synonyms
        assert text_synonyms == response_synonyms


class TestNormalizeTaskOutput:
    """Tests for normalize_task_output function"""

    def test_normalize_text_to_completion(self):
        """Test normalizing 'text' to 'completion'"""
        task_output = {"text": "hello world"}
        required = ["completion"]
        normalized = normalize_task_output(task_output, required)

        assert "completion" in normalized
        assert normalized["completion"] == "hello world"
        assert "text" not in normalized  # Original field removed after mapping

    def test_normalize_answer_to_completion(self):
        """Test normalizing 'answer' to 'completion'"""
        task_output = {"answer": "Paris"}
        required = ["completion"]
        normalized = normalize_task_output(task_output, required)

        assert "completion" in normalized
        assert normalized["completion"] == "Paris"

    def test_normalize_prompt_to_question(self):
        """Test normalizing 'prompt' to 'question'"""
        task_output = {"prompt": "What is the capital?"}
        required = ["question"]
        normalized = normalize_task_output(task_output, required)

        assert "question" in normalized
        assert normalized["question"] == "What is the capital?"

    def test_normalize_context_to_reference(self):
        """Test normalizing 'context' to 'reference'"""
        task_output = {"context": "The sky is blue"}
        required = ["reference"]
        normalized = normalize_task_output(task_output, required)

        assert "reference" in normalized
        assert normalized["reference"] == "The sky is blue"

    def test_normalize_ground_truth_to_reference(self):
        """Test normalizing 'ground_truth' to 'reference'"""
        task_output = {"ground_truth": "Correct answer"}
        required = ["reference"]
        normalized = normalize_task_output(task_output, required)

        assert "reference" in normalized
        assert normalized["reference"] == "Correct answer"

    def test_normalize_prompts_to_trajectory_prompts(self):
        """Test normalizing 'prompts' to 'trajectory_prompts'"""
        task_output = {"prompts": "prompt1, prompt2"}
        required = ["trajectory_prompts"]
        normalized = normalize_task_output(task_output, required)

        assert "trajectory_prompts" in normalized
        assert normalized["trajectory_prompts"] == "prompt1, prompt2"

    def test_normalize_completions_to_trajectory_completions(self):
        """Test normalizing 'completions' to 'trajectory_completions'"""
        task_output = {"completions": "comp1, comp2"}
        required = ["trajectory_completions"]
        normalized = normalize_task_output(task_output, required)

        assert "trajectory_completions" in normalized
        assert normalized["trajectory_completions"] == "comp1, comp2"

    def test_normalize_multiple_fields(self):
        """Test normalizing multiple fields at once"""
        task_output = {
            "answer": "Paris",
            "prompt": "What is the capital?",
            "context": "France",
        }
        required = ["completion", "question", "reference"]
        normalized = normalize_task_output(task_output, required)

        assert normalized["completion"] == "Paris"
        assert normalized["question"] == "What is the capital?"
        assert normalized["reference"] == "France"

    def test_normalize_with_no_mapping_needed(self):
        """Test when field names already match required fields"""
        task_output = {"completion": "hello", "question": "greet"}
        required = ["completion", "question"]
        normalized = normalize_task_output(task_output, required)

        assert normalized["completion"] == "hello"
        assert normalized["question"] == "greet"

    def test_normalize_preserves_extra_fields(self):
        """Test that extra fields not in required list are preserved"""
        task_output = {
            "answer": "Paris",
            "extra_field": "value",
            "another": 123,
        }
        required = ["completion"]
        normalized = normalize_task_output(task_output, required)

        assert normalized["completion"] == "Paris"
        assert "extra_field" in normalized
        assert "another" in normalized
        assert normalized["extra_field"] == "value"
        assert normalized["another"] == 123

    def test_normalize_empty_task_output(self):
        """Test normalizing with empty task output"""
        task_output = {}
        required = ["completion"]
        normalized = normalize_task_output(task_output, required)

        assert "completion" not in normalized
        assert len(normalized) == 0

    def test_normalize_empty_required_fields(self):
        """Test normalizing with empty required fields"""
        task_output = {"text": "hello", "extra": "value"}
        required = []
        normalized = normalize_task_output(task_output, required)

        # Should preserve all fields when no mapping needed
        assert "text" in normalized
        assert "extra" in normalized

    def test_normalize_prioritizes_exact_match(self):
        """Test that exact field match is preferred over synonyms"""
        # If both synonym and exact match exist, exact match should be used
        task_output = {
            "completion": "exact",
            "answer": "synonym",
        }
        required = ["completion"]
        normalized = normalize_task_output(task_output, required)

        # Should use the exact match "completion"
        assert normalized["completion"] == "exact"

    def test_normalize_with_non_synonym_fields(self):
        """Test normalizing fields that have no synonyms"""
        task_output = {
            "tool_input": "input_value",
            "tool_output": "output_value",
        }
        required = ["tool_input", "tool_output"]
        normalized = normalize_task_output(task_output, required)

        assert normalized["tool_input"] == "input_value"
        assert normalized["tool_output"] == "output_value"

    def test_normalize_mixed_synonyms_and_non_synonyms(self):
        """Test normalizing a mix of synonym and non-synonym fields"""
        task_output = {
            "answer": "Paris",
            "tool_input": "search",
            "prompt": "What is the capital?",
        }
        required = ["completion", "tool_input", "question"]
        normalized = normalize_task_output(task_output, required)

        assert normalized["completion"] == "Paris"
        assert normalized["tool_input"] == "search"
        assert normalized["question"] == "What is the capital?"


class TestGetFieldSuggestions:
    """Tests for get_field_suggestions function"""

    def test_suggest_synonym_for_missing_completion(self):
        """Test suggesting synonyms for missing 'completion' field"""
        missing = "completion"
        available = ["answer", "question"]
        suggestions = get_field_suggestions(missing, available)

        assert "answer" in suggestions
        assert "question" not in suggestions

    def test_suggest_synonym_for_missing_question(self):
        """Test suggesting synonyms for missing 'question' field"""
        missing = "question"
        available = ["prompt", "text"]
        suggestions = get_field_suggestions(missing, available)

        assert "prompt" in suggestions
        assert "text" not in suggestions

    def test_suggest_synonym_for_missing_reference(self):
        """Test suggesting synonyms for missing 'reference' field"""
        missing = "reference"
        available = ["context", "completion"]
        suggestions = get_field_suggestions(missing, available)

        assert "context" in suggestions
        assert "completion" not in suggestions

    def test_no_suggestions_when_no_synonyms_available(self):
        """Test no suggestions when available fields have no synonyms"""
        missing = "completion"
        available = ["tool_input", "tool_output"]
        suggestions = get_field_suggestions(missing, available)

        assert len(suggestions) == 0

    def test_multiple_synonyms_available(self):
        """Test suggesting when multiple synonyms are available"""
        missing = "completion"
        available = ["answer", "text", "response"]
        suggestions = get_field_suggestions(missing, available)

        assert "answer" in suggestions
        assert "text" in suggestions
        assert "response" in suggestions

    def test_empty_available_fields(self):
        """Test with empty available fields"""
        missing = "completion"
        available = []
        suggestions = get_field_suggestions(missing, available)

        assert len(suggestions) == 0


class TestFormatFieldHelp:
    """Tests for format_field_help function"""

    def test_format_field_with_synonyms(self):
        """Test formatting help for field with synonyms"""
        help_text = format_field_help("completion")
        assert "completion" in help_text
        assert "answer" in help_text
        assert "response" in help_text
        assert "text" in help_text
        assert "synonym" in help_text.lower()

    def test_format_field_without_synonyms(self):
        """Test formatting help for field without synonyms"""
        help_text = format_field_help("tool_input")
        assert help_text == "'tool_input'"
        assert "synonym" not in help_text.lower()

    def test_format_multiple_fields(self):
        """Test formatting help for multiple fields"""
        fields = ["completion", "question", "reference"]
        help_texts = [format_field_help(field) for field in fields]

        assert len(help_texts) == 3
        assert all("synonym" in text.lower() for text in help_texts)


class TestIntegrationWithValidateTaskOutput:
    """Integration tests with validate_and_normalize_task_output"""

    def test_validate_with_synonym_mapping(self):
        """Test that validate_and_normalize_task_output uses synonym mapping"""
        from traceloop.sdk.evaluator.evaluator import validate_and_normalize_task_output
        from traceloop.sdk.evaluator.config import EvaluatorDetails

        # User returns "answer" but evaluator needs "completion"
        task_output = {"answer": "Paris", "prompt": "What is the capital?"}
        evaluators = [
            EvaluatorDetails(
                slug="test-evaluator",
                required_input_fields=["completion", "question"],
            )
        ]

        # Should not raise - synonyms should be mapped
        normalized = validate_and_normalize_task_output(task_output, evaluators)
        assert "completion" in normalized
        assert "question" in normalized
        assert normalized["completion"] == "Paris"
        assert normalized["question"] == "What is the capital?"

    def test_validate_fails_with_helpful_message(self):
        """Test that validation failure includes synonym suggestions"""
        from traceloop.sdk.evaluator.evaluator import validate_and_normalize_task_output
        from traceloop.sdk.evaluator.config import EvaluatorDetails

        task_output = {"wrong_field": "value"}
        evaluators = [
            EvaluatorDetails(
                slug="test-evaluator",
                required_input_fields=["completion"],
            )
        ]

        with pytest.raises(ValueError) as exc_info:
            validate_and_normalize_task_output(task_output, evaluators)

        error_message = str(exc_info.value)
        assert "test-evaluator requires:" in error_message
        assert "completion" in error_message
        assert "synonym" in error_message.lower()

    def test_validate_with_context_to_reference_mapping(self):
        """Test specific case of context mapping to reference"""
        from traceloop.sdk.evaluator.evaluator import validate_and_normalize_task_output
        from traceloop.sdk.evaluator.config import EvaluatorDetails

        task_output = {
            "answer": "Yes",
            "question": "Is it true?",
            "context": "The sky is blue",
        }
        evaluators = [
            EvaluatorDetails(
                slug="faithfulness",
                required_input_fields=["completion", "question", "reference"],
            )
        ]

        normalized = validate_and_normalize_task_output(task_output, evaluators)
        assert normalized["completion"] == "Yes"
        assert normalized["question"] == "Is it true?"
        assert normalized["reference"] == "The sky is blue"

    def test_validate_with_trajectory_fields(self):
        """Test mapping for trajectory fields used in agent evaluators"""
        from traceloop.sdk.evaluator.evaluator import validate_and_normalize_task_output
        from traceloop.sdk.evaluator.config import EvaluatorDetails

        task_output = {
            "prompts": "prompt1, prompt2",
            "completions": "comp1, comp2",
        }
        evaluators = [
            EvaluatorDetails(
                slug="agent-efficiency",
                required_input_fields=["trajectory_prompts", "trajectory_completions"],
            )
        ]

        normalized = validate_and_normalize_task_output(task_output, evaluators)
        assert normalized["trajectory_prompts"] == "prompt1, prompt2"
        assert normalized["trajectory_completions"] == "comp1, comp2"
