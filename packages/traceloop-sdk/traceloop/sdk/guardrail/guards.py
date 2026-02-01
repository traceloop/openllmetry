"""
Simplified guard factory with sensible defaults.

Provides pre-configured guards ready for use with client.guardrails.create().

Example:
    from traceloop.sdk import Traceloop
    from traceloop.sdk.guardrail import toxicity_guard, pii_guard, answer_relevancy_guard, OnFailure

    client = Traceloop.init(api_key="...")

    guardrail = client.guardrails.create(
        guards=[
            toxicity_guard(),
            pii_guard(),
            answer_relevancy_guard(),
        ],
        on_failure=OnFailure.raise_exception("Content policy violation"),
    )

    result = await guardrail.run(generate_content)
"""

from __future__ import annotations

from typing import Any, Callable, Awaitable

from .condition import Condition
from ..generated.evaluators.definitions import EvaluatorMadeByTraceloop
from ..evaluator.config import EvaluatorDetails

# Type alias for guard functions
Guard = Callable[[Any], Awaitable[bool]]


def _create_guard(
    evaluator_details: EvaluatorDetails,
    condition: Callable[[Any], bool],
    timeout_in_sec: int = 60,
) -> Guard:
    """
    Convert an EvaluatorDetails to a guard function.

    Args:
        evaluator_details: The evaluator configuration
        condition: Function that receives evaluator result and returns bool.
                   True = pass, False = fail.
        timeout_in_sec: Maximum time to wait for evaluator execution

    Returns:
        Async function suitable for client.guardrails.create(guards=[...])
    """

    evaluator_slug = evaluator_details.slug
    evaluator_version = evaluator_details.version
    evaluator_config = evaluator_details.config
    condition_field = evaluator_details.condition_field

    async def guard_fn(input_data: Any) -> bool:
        from traceloop.sdk import Traceloop
        from traceloop.sdk.evaluator.evaluator import Evaluator

        # Convert Pydantic model to dict, or use dict directly
        if isinstance(input_data, dict):
            input_dict = input_data
        elif hasattr(input_data, "model_dump"):
            input_dict = input_data.model_dump()
        else:
            input_dict = dict(input_data)

        client = Traceloop.get()
        evaluator = Evaluator(client._async_http)

        eval_response = await evaluator.run(
            evaluator_slug=evaluator_slug,
            input=input_dict,
            evaluator_version=evaluator_version,
            evaluator_config=evaluator_config,
            timeout_in_sec=timeout_in_sec,
        )

        if condition_field:
            result_to_validate = eval_response.result.evaluator_result[condition_field]
        else:
            result_to_validate = eval_response.result.evaluator_result

        return condition(result_to_validate)

    guard_fn.__name__ = evaluator_slug
    return guard_fn


def custom_evaluator_guard(
    evaluator_slug: str,
    evaluator_version: str | None = None,
    evaluator_config: dict[str, Any] | None = None,
    condition_field: str = "pass",
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when the custom evaluator passes."""
    evaluator_details = EvaluatorDetails(
        slug=evaluator_slug,
        condition_field=condition_field,
        version=evaluator_version,
        config=evaluator_config,
    )
    return _create_guard(evaluator_details, condition, timeout_in_sec)


# =============================================================================
# Safety Detectors - pass when content is safe (is_safe = True)
# =============================================================================


def toxicity_guard(
    threshold: float | None = None,
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when content is safe from toxicity."""
    evaluator = (
        EvaluatorMadeByTraceloop.toxicity_detector(threshold=threshold)
        if threshold is not None
        else EvaluatorMadeByTraceloop.toxicity_detector()
    )
    return _create_guard(
        evaluator,
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def profanity_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when content is free from profanity."""
    return _create_guard(
        EvaluatorMadeByTraceloop.profanity_detector(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def sexism_guard(
    threshold: float | None = None,
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when content is free from sexism."""
    evaluator = (
        EvaluatorMadeByTraceloop.sexism_detector(threshold=threshold)
        if threshold is not None
        else EvaluatorMadeByTraceloop.sexism_detector()
    )
    return _create_guard(
        evaluator,
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


# =============================================================================
# Detection Guards - pass when NOT detected (has_* = False)
# =============================================================================


def pii_guard(
    probability_threshold: float | None = None,
    condition: Callable[[Any], bool] = Condition.is_false(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when no PII is detected."""
    evaluator = (
        EvaluatorMadeByTraceloop.pii_detector(
            probability_threshold=probability_threshold
        )
        if probability_threshold is not None
        else EvaluatorMadeByTraceloop.pii_detector()
    )
    return _create_guard(
        evaluator,
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def secrets_guard(
    condition: Callable[[Any], bool] = Condition.is_false(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when no secrets are detected."""
    return _create_guard(
        EvaluatorMadeByTraceloop.secrets_detector(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def prompt_injection_guard(
    threshold: float | None = None,
    condition: Callable[[Any], bool] = Condition.is_false(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when no prompt injection is detected."""
    evaluator = (
        EvaluatorMadeByTraceloop.prompt_injection(threshold=threshold)
        if threshold is not None
        else EvaluatorMadeByTraceloop.prompt_injection()
    )
    return _create_guard(
        evaluator,
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


# =============================================================================
# Quality Evaluators - pass when quality is good (is_* = True)
# =============================================================================


def answer_relevancy_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when the answer is relevant to the question."""
    return _create_guard(
        EvaluatorMadeByTraceloop.answer_relevancy(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def faithfulness_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when the response is faithful to the context."""
    return _create_guard(
        EvaluatorMadeByTraceloop.faithfulness(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


# =============================================================================
# Validators - pass when valid (is_valid_* = True)
# =============================================================================


def json_validator_guard(
    enable_schema_validation: bool | None = None,
    schema_string: str | None = None,
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when the text is valid JSON."""
    return _create_guard(
        EvaluatorMadeByTraceloop.json_validator(
            enable_schema_validation=enable_schema_validation,
            schema_string=schema_string,
        ),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def sql_validator_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when the text is valid SQL."""
    return _create_guard(
        EvaluatorMadeByTraceloop.sql_validator(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def regex_validator_guard(
    regex: str | None = None,
    case_sensitive: bool | None = None,
    dot_include_nl: bool | None = None,
    multi_line: bool | None = None,
    should_match: bool | None = None,
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when the text matches the regex pattern."""
    return _create_guard(
        EvaluatorMadeByTraceloop.regex_validator(
            case_sensitive=case_sensitive,
            dot_include_nl=dot_include_nl,
            multi_line=multi_line,
            regex=regex,
            should_match=should_match,
        ),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def placeholder_regex_guard(
    case_sensitive: bool | None = None,
    dot_include_nl: bool | None = None,
    multi_line: bool | None = None,
    should_match: bool | None = None,
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when placeholder regex validation succeeds."""
    return _create_guard(
        EvaluatorMadeByTraceloop.placeholder_regex(
            case_sensitive=case_sensitive,
            dot_include_nl=dot_include_nl,
            multi_line=multi_line,
            should_match=should_match,
        ),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


# =============================================================================
# Agent Evaluators - pass on success
# =============================================================================


def agent_efficiency_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when agent efficiency score meets threshold."""
    return _create_guard(
        EvaluatorMadeByTraceloop.agent_efficiency(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def agent_flow_quality_guard(
    conditions: list[str],
    threshold: float,
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when agent flow meets quality conditions."""
    return _create_guard(
        EvaluatorMadeByTraceloop.agent_flow_quality(
            conditions=conditions,
            threshold=threshold,
        ),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def agent_goal_accuracy_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when agent goal accuracy score meets threshold."""
    return _create_guard(
        EvaluatorMadeByTraceloop.agent_goal_accuracy(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def agent_goal_completeness_guard(
    threshold: float,
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when agent completes its goal."""
    return _create_guard(
        EvaluatorMadeByTraceloop.agent_goal_completeness(threshold=threshold),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def agent_tool_error_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when no tool errors are detected."""
    return _create_guard(
        EvaluatorMadeByTraceloop.agent_tool_error_detector(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def agent_tool_trajectory_guard(
    input_params_sensitive: bool | None = None,
    mismatch_sensitive: bool | None = None,
    order_sensitive: bool | None = None,
    threshold: float | None = None,
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when agent tool trajectory matches expected."""
    return _create_guard(
        EvaluatorMadeByTraceloop.agent_tool_trajectory(
            input_params_sensitive=input_params_sensitive,
            mismatch_sensitive=mismatch_sensitive,
            order_sensitive=order_sensitive,
            threshold=threshold,
        ),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def intent_change_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when intent remains consistent."""
    return _create_guard(
        EvaluatorMadeByTraceloop.intent_change(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


# =============================================================================
# Score-based Evaluators - pass when score >= threshold (default 0.8)
# =============================================================================


def answer_correctness_guard(
    threshold: float = 0.8,
    condition: Callable[[Any], bool] | None = None,
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when answer correctness score meets threshold."""
    effective_condition = (
        condition if condition is not None else Condition.greater_than_or_equal(threshold)
    )
    return _create_guard(
        EvaluatorMadeByTraceloop.answer_correctness(),
        condition=effective_condition,
        timeout_in_sec=timeout_in_sec,
    )


def answer_completeness_guard(
    threshold: float = 0.8,
    condition: Callable[[Any], bool] | None = None,
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when answer completeness score meets threshold."""
    effective_condition = (
        condition if condition is not None else Condition.greater_than_or_equal(threshold)
    )
    return _create_guard(
        EvaluatorMadeByTraceloop.answer_completeness(),
        condition=effective_condition,
        timeout_in_sec=timeout_in_sec,
    )


def context_relevance_guard(
    threshold: float = 0.8,
    model: str | None = None,
    condition: Callable[[Any], bool] | None = None,
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when context relevance score meets threshold."""
    effective_condition = (
        condition if condition is not None else Condition.greater_than_or_equal(threshold)
    )
    return _create_guard(
        EvaluatorMadeByTraceloop.context_relevance(model=model),
        condition=effective_condition,
        timeout_in_sec=timeout_in_sec,
    )


def conversation_quality_guard(
    threshold: float = 0.7,
    condition: Callable[[Any], bool] | None = None,
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when conversation quality score meets threshold."""
    effective_condition = (
        condition if condition is not None else Condition.greater_than_or_equal(threshold)
    )
    return _create_guard(
        EvaluatorMadeByTraceloop.conversation_quality(),
        condition=effective_condition,
        timeout_in_sec=timeout_in_sec,
    )


def html_comparison_guard(
    threshold: float = 0.9,
    condition: Callable[[Any], bool] | None = None,
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when HTML similarity score meets threshold."""
    effective_condition = (
        condition if condition is not None else Condition.greater_than_or_equal(threshold)
    )
    return _create_guard(
        EvaluatorMadeByTraceloop.html_comparison(),
        condition=effective_condition,
        timeout_in_sec=timeout_in_sec,
    )


def instruction_adherence_guard(
    threshold: float = 0.8,
    condition: Callable[[Any], bool] | None = None,
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when instruction adherence score meets threshold."""
    effective_condition = (
        condition if condition is not None else Condition.greater_than_or_equal(threshold)
    )
    return _create_guard(
        EvaluatorMadeByTraceloop.instruction_adherence(),
        condition=effective_condition,
        timeout_in_sec=timeout_in_sec,
    )


def semantic_similarity_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when semantic similarity score meets threshold."""
    return _create_guard(
        EvaluatorMadeByTraceloop.semantic_similarity(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def topic_adherence_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when topic adherence score meets threshold."""
    return _create_guard(
        EvaluatorMadeByTraceloop.topic_adherence(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


# =============================================================================
# Score-based Evaluators (lower is better) - pass when score <= threshold
# =============================================================================


def perplexity_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when perplexity score is below threshold."""
    return _create_guard(
        EvaluatorMadeByTraceloop.perplexity(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def prompt_perplexity_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when prompt perplexity score is below threshold."""
    return _create_guard(
        EvaluatorMadeByTraceloop.prompt_perplexity(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def uncertainty_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when uncertainty is below threshold."""
    return _create_guard(
        EvaluatorMadeByTraceloop.uncertainty_detector(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


# =============================================================================
# Count/Ratio Evaluators - require explicit min/max bounds
# =============================================================================


def word_count_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when word count is within specified bounds."""
    return _create_guard(
        EvaluatorMadeByTraceloop.word_count(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def char_count_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when character count is within specified bounds."""
    return _create_guard(
        EvaluatorMadeByTraceloop.char_count(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def word_count_ratio_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when word count ratio is within specified bounds."""
    return _create_guard(
        EvaluatorMadeByTraceloop.word_count_ratio(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def char_count_ratio_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when character count ratio is within specified bounds."""
    return _create_guard(
        EvaluatorMadeByTraceloop.char_count_ratio(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


# =============================================================================
# Special Evaluators
# =============================================================================


def tone_detection_guard(
    condition: Callable[[Any], bool] = Condition.is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when detected tone matches expected."""
    return _create_guard(
        EvaluatorMadeByTraceloop.tone_detection(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )
