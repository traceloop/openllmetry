"""
Unit tests for guardrail _validate_inputs method.

Tests both pass and fail cases for length and type validation.
"""
import pytest
from typing import Union
from unittest.mock import MagicMock
from pydantic import BaseModel

from traceloop.sdk.guardrail.guardrail import Guardrails
from traceloop.sdk.guardrail.model import GuardInputTypeError
from traceloop.sdk.guardrail import Condition
from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop
from traceloop.sdk.generated.evaluators.request import (
    PIIDetectorInput,
    ToxicityDetectorInput,
    AnswerRelevancyInput,
)


class MockInput(BaseModel):
    """Pydantic model for testing type validation."""
    text: str
    score: float


class AnotherInput(BaseModel):
    """Different Pydantic model for testing type mismatches."""
    name: str
    count: int


def create_guardrails_with_guards(guards: list) -> Guardrails:
    """Helper to create a Guardrails instance with specified guards."""
    mock_client = MagicMock()
    guardrails = Guardrails(mock_client)
    guardrails._guards = guards
    guardrails._on_failure = lambda x: None
    return guardrails


class TestValidateInputsPass:
    """Tests for _validate_inputs that should pass."""

    def test_lambda_guards(self):
        """Lambda guards should validate correctly."""
        guards = [
            lambda z: z["score"] > 0.5,
            lambda z: "bad" not in z["text"],
        ]
        guardrails = create_guardrails_with_guards(guards)

        # Any types work since lambdas have no annotations
        guard_inputs = [
            {"score": 0.8},
            {"text": "hello"},
        ]
        guardrails._validate_inputs(guard_inputs)

    def test_custom_function_as_a_guard(self):
        """Annotated function with dict type should validate correctly."""
        def guard(data: dict) -> bool:
            return data.get("score", 0) > 0.5

        guardrails = create_guardrails_with_guards([guard])
        guardrails._validate_inputs([{"score": 0.8}])

    def test_custom_function_with_correct_pydantic_model(self):
        """Annotated function with Pydantic model should validate correctly."""
        def guard(data: MockInput) -> bool:
            return data.score > 0.5

        guardrails = create_guardrails_with_guards([guard])
        guard_input = MockInput(text="hello", score=0.8)
        guardrails._validate_inputs([guard_input])

    def test_pydantic_model_from_dict(self):
        """Pydantic TypeAdapter can coerce dict to model."""
        def guard(data: MockInput) -> bool:
            return data.score > 0.5

        guardrails = create_guardrails_with_guards([guard])
        # Dict that matches MockInput schema should be coerced
        guardrails._validate_inputs([{"text": "hello", "score": 0.8}])

    def test_mixed_guards_lambda_and_annotated(self):
        """Mixed lambda and annotated guards with correct types."""
        def typed_guard(data: dict) -> bool:
            return data.get("valid", False)

        guards = [
            lambda z: z > 0,  # Lambda, no type check
            typed_guard,       # Annotated, expects dict
        ]
        guardrails = create_guardrails_with_guards(guards)

        guard_inputs = [
            42,              # Any type for lambda
            {"valid": True}, # Dict for typed guard
        ]
        guardrails._validate_inputs(guard_inputs)  # Should not raise

    def test_multiple_annotated_guards_correct_types(self):
        """Multiple annotated guards with different correct types."""
        def guard1(data: dict) -> bool:
            return True

        def guard2(data: MockInput) -> bool:
            return True

        def guard3(data: str) -> bool:
            return True

        guards = [guard1, guard2, guard3]
        guardrails = create_guardrails_with_guards(guards)

        guard_inputs = [
            {"key": "value"},
            MockInput(text="hello", score=0.5),
            "some string",
        ]
        guardrails._validate_inputs(guard_inputs)  # Should not raise

    def test_traceloop_pii_detector_input(self):
        """Guard with PIIDetectorInput type annotation should validate correctly."""

        guardrails = create_guardrails_with_guards(
            [EvaluatorMadeByTraceloop.pii_detector().as_guard(condition=Condition.is_false(field="has_pii"))])

        # With Pydantic model instance
        guardrails._validate_inputs([PIIDetectorInput(text="Hello world")])

        # With dict that matches schema (Pydantic coerces it)
        guardrails._validate_inputs([{"text": "Hello world"}])

    def test_traceloop_toxicity_detector_input(self):
        """Guard using EvaluatorMadeByTraceloop.toxicity_detector() should validate correctly."""
        guardrails = create_guardrails_with_guards([
            EvaluatorMadeByTraceloop.toxicity_detector().as_guard(
                condition=Condition.is_false(field="is_toxic"),
                timeout_in_sec=30,
            )
        ])

        # With Pydantic model instance
        guardrails._validate_inputs([ToxicityDetectorInput(text="Friendly message")])

        # With dict that matches schema
        guardrails._validate_inputs([{"text": "Friendly message"}])

    def test_traceloop_answer_relevancy_input(self):
        """Guard with AnswerRelevancyInput type annotation should validate correctly."""
        guardrails = create_guardrails_with_guards([
            EvaluatorMadeByTraceloop.answer_relevancy().as_guard(
                condition=Condition.is_false(field="is_relevant"),
                timeout_in_sec=30,
            )
        ])

        # With Pydantic model instance
        guard_input = AnswerRelevancyInput(
            answer="Paris is the capital of France.",
            question="What is the capital of France?",
        )
        guardrails._validate_inputs([guard_input])

        # With dict that matches schema
        guardrails._validate_inputs([{
            "answer": "Paris is the capital of France.",
            "question": "What is the capital of France?",
        }])


    def test_multiple_traceloop_evaluator_guards(self):
        """Multiple guards using EvaluatorMadeByTraceloop evaluators."""
        guards = [
            EvaluatorMadeByTraceloop.pii_detector().as_guard(
                condition=Condition.is_false(field="has_pii"),
                timeout_in_sec=30,
            ),
            EvaluatorMadeByTraceloop.toxicity_detector().as_guard(
                condition=Condition.is_false(field="is_toxic"),
                timeout_in_sec=30,
            ),
        ]
        guardrails = create_guardrails_with_guards(guards)

        guard_inputs = [
            PIIDetectorInput(text="Check for PII"),
            ToxicityDetectorInput(text="Check for toxicity"),
        ]
        guardrails._validate_inputs(guard_inputs)

    def test_mixed_lambda_and_traceloop_evaluator_guards(self):
        """Mix of lambda guards and EvaluatorMadeByTraceloop evaluator guards."""
        guards = [
            lambda z: z["score"] > 0.5,  # Lambda guard
            EvaluatorMadeByTraceloop.pii_detector().as_guard(
                condition=Condition.is_false(field="has_pii"),
                timeout_in_sec=30,
            ),
        ]
        guardrails = create_guardrails_with_guards(guards)

        guard_inputs = [
            {"score": 0.8},
            PIIDetectorInput(text="Hello"),
        ]
        guardrails._validate_inputs(guard_inputs)


class TestValidateInputsFail:
    """Tests for _validate_inputs that should fail."""

    def test_length_mismatch_fewer_inputs(self):
        """Fewer guard_inputs than guards raises ValueError."""
        guards = [
            lambda z: True,
            lambda z: True,
            lambda z: True,
        ]
        guardrails = create_guardrails_with_guards(guards)

        with pytest.raises(ValueError) as exc_info:
            guardrails._validate_inputs([{"a": 1}])  # Only 1 input for 3 guards

        assert "Number of guard_inputs (1)" in str(exc_info.value)
        assert "must match number of guards (3)" in str(exc_info.value)

    def test_length_mismatch_more_inputs(self):
        """More guard_inputs than guards raises ValueError."""
        guards = [lambda z: True]
        guardrails = create_guardrails_with_guards(guards)

        with pytest.raises(ValueError) as exc_info:
            guardrails._validate_inputs([{"a": 1}, {"b": 2}, {"c": 3}])

        assert "Number of guard_inputs (3)" in str(exc_info.value)
        assert "must match number of guards (1)" in str(exc_info.value)

    def test_type_mismatch_expected_dict_got_string(self):
        """Type mismatch: expected dict, got string raises GuardInputTypeError."""
        def guard(data: dict) -> bool:
            return True

        guardrails = create_guardrails_with_guards([guard])

        with pytest.raises(GuardInputTypeError) as exc_info:
            guardrails._validate_inputs(["not a dict"])

        assert exc_info.value.guard_index == 0
        assert exc_info.value.expected_type is dict
        assert exc_info.value.actual_type is str
        assert exc_info.value.validation_error is not None

    def test_type_mismatch_pydantic_model_wrong_fields(self):
        """Pydantic model with missing required fields raises GuardInputTypeError."""
        def guard(data: MockInput) -> bool:
            return True

        guardrails = create_guardrails_with_guards([guard])

        with pytest.raises(GuardInputTypeError) as exc_info:
            # Missing 'score' field
            guardrails._validate_inputs([{"text": "hello"}])

        assert exc_info.value.guard_index == 0
        assert exc_info.value.expected_type is MockInput

    def test_type_mismatch_on_second_guard(self):
        """Type mismatch on second guard reports correct index."""
        def guard1(data: dict) -> bool:
            return True

        def guard2(data: MockInput) -> bool:
            return True

        guards = [guard1, guard2]
        guardrails = create_guardrails_with_guards(guards)

        with pytest.raises(GuardInputTypeError) as exc_info:
            guardrails._validate_inputs([
                {"valid": True},  # Correct for guard1
                "wrong type",     # Wrong for guard2
            ])

        assert exc_info.value.guard_index == 1
        assert exc_info.value.expected_type is MockInput
        assert exc_info.value.actual_type is str

    def test_type_mismatch_expected_string_got_int(self):
        """Type mismatch: expected str, got int raises GuardInputTypeError."""
        def guard(data: str) -> bool:
            return len(data) > 0

        guardrails = create_guardrails_with_guards([guard])

        with pytest.raises(GuardInputTypeError) as exc_info:
            guardrails._validate_inputs([12345])

        assert exc_info.value.guard_index == 0
        assert exc_info.value.expected_type is str
        assert exc_info.value.actual_type is int

    def test_empty_guard_inputs_with_guards(self):
        """Empty guard_inputs with guards raises ValueError."""
        guards = [lambda z: True, lambda z: True]
        guardrails = create_guardrails_with_guards(guards)

        with pytest.raises(ValueError) as exc_info:
            guardrails._validate_inputs([])

        assert "Number of guard_inputs (0)" in str(exc_info.value)
        assert "must match number of guards (2)" in str(exc_info.value)

    def test_traceloop_evaluator_length_mismatch_fewer_inputs(self):
        """Traceloop evaluator guards with fewer inputs raises ValueError."""
        guards = [
            EvaluatorMadeByTraceloop.pii_detector().as_guard(
                condition=Condition.is_false(field="has_pii"),
            ),
            EvaluatorMadeByTraceloop.toxicity_detector().as_guard(
                condition=Condition.is_false(field="is_toxic"),
            ),
        ]
        guardrails = create_guardrails_with_guards(guards)

        with pytest.raises(ValueError) as exc_info:
            # Only 1 input for 2 guards
            guardrails._validate_inputs([PIIDetectorInput(text="Hello")])

        assert "Number of guard_inputs (1)" in str(exc_info.value)
        assert "must match number of guards (2)" in str(exc_info.value)

    def test_traceloop_evaluator_length_mismatch_more_inputs(self):
        """Traceloop evaluator guards with more inputs raises ValueError."""
        guards = [
            EvaluatorMadeByTraceloop.pii_detector().as_guard(
                condition=Condition.is_false(field="has_pii"),
            ),
        ]
        guardrails = create_guardrails_with_guards(guards)

        with pytest.raises(ValueError) as exc_info:
            # 3 inputs for 1 guard
            guardrails._validate_inputs([
                PIIDetectorInput(text="Hello"),
                ToxicityDetectorInput(text="World"),
                {"extra": "input"},
            ])

        assert "Number of guard_inputs (3)" in str(exc_info.value)
        assert "must match number of guards (1)" in str(exc_info.value)

    def test_mixed_evaluator_and_typed_guard_type_mismatch(self):
        """Mixed evaluator + typed guard where typed guard fails type check."""
        def typed_guard(data: PIIDetectorInput) -> bool:
            return True

        guards = [
            EvaluatorMadeByTraceloop.toxicity_detector().as_guard(
                condition=Condition.is_false(field="is_toxic"),
            ),
            typed_guard,  # This has type annotation
        ]
        guardrails = create_guardrails_with_guards(guards)

        with pytest.raises(GuardInputTypeError) as exc_info:
            guardrails._validate_inputs([
                ToxicityDetectorInput(text="Hello"),  # Correct for evaluator guard
                "wrong type",  # Wrong for typed_guard (expects PIIDetectorInput)
            ])

        assert exc_info.value.guard_index == 1
        assert exc_info.value.expected_type is PIIDetectorInput
        assert exc_info.value.actual_type is str

    def test_guard_inputs_wrong_order(self):
        """Guard inputs in wrong order should fail type validation."""

        # Guards expect: [PIIDetectorInput, ToxicityDetectorInput]
        guards = [
            EvaluatorMadeByTraceloop.pii_detector().as_guard(
                condition=Condition.is_false(field="has_pii"),
            ),
            EvaluatorMadeByTraceloop.toxicity_detector().as_guard(
                condition=Condition.is_false(field="is_toxic"),
            ),
        ]
        guardrails = create_guardrails_with_guards(guards)

        # Inputs provided in WRONG order: [ToxicityDetectorInput, PIIDetectorInput]
        with pytest.raises(GuardInputTypeError) as exc_info:
            guardrails._validate_inputs([
                ToxicityDetectorInput(text="Should be second"),  # Wrong! Expected PIIDetectorInput
                PIIDetectorInput(text="Should be first"),        # Wrong! Expected ToxicityDetectorInput
            ])

        # Should fail on first guard (index 0) since it expected PIIDetectorInput
        assert exc_info.value.guard_index == 0
        assert exc_info.value.expected_type is PIIDetectorInput
        assert exc_info.value.actual_type is ToxicityDetectorInput


class TestValidateInputsEdgeCases:
    """Edge cases for _validate_inputs."""

    def test_empty_guards_and_empty_inputs(self):
        """Empty guards with empty inputs should pass."""
        guardrails = create_guardrails_with_guards([])
        guardrails._validate_inputs([])  # Should not raise

    def test_guard_with_no_parameters(self):
        """Guard with no parameters is skipped."""
        def guard() -> bool:
            return True

        guardrails = create_guardrails_with_guards([guard])
        # This guard has no parameters, so any input passes length check
        # but type validation is skipped
        guardrails._validate_inputs(["anything"])  # Should not raise

    def test_async_guard_function(self):
        """Async guard functions should have their types validated."""
        async def guard(data: dict) -> bool:
            return True

        guardrails = create_guardrails_with_guards([guard])
        guardrails._validate_inputs([{"key": "value"}])  # Should not raise

        with pytest.raises(GuardInputTypeError):
            guardrails._validate_inputs(["not a dict"])
