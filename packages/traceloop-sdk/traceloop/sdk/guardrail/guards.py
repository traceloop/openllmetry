"""
Simplified guard factory with sensible defaults.

Provides pre-configured guards ready for use with Guardrails().

Example:
    from traceloop.sdk.guardrail import Guardrails, toxicity_guard, pii_guard, answer_relevancy_guard

    guardrail = Guardrails(
        toxicity_guard(),
        pii_guard(),
        answer_relevancy_guard(),
        on_failure="raise",
    )

    result = await guardrail.run(generate_content)
"""

from __future__ import annotations

import json
from typing import Any, Callable, Awaitable

from opentelemetry import trace

from .conditions import is_true, is_false
from .span_attributes import GEN_AI_GUARDRAIL_OUTPUT
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
        Async function suitable for client.create_guardrail(guards=[...])
    """

    evaluator_slug = evaluator_details.slug
    evaluator_config = evaluator_details.config
    input_schema = evaluator_details.input_schema
    condition_field = evaluator_details.condition_field

    async def guard_fn(input_data: Any) -> bool:
        from traceloop.sdk import Traceloop
        from traceloop.sdk.guardrail.guardrail import Guardrails

        # Convert Pydantic model to dict, or use dict directly
        if isinstance(input_data, dict):
            input_dict = input_data
        elif hasattr(input_data, "model_dump"):
            input_dict = input_data.model_dump()
        else:
            input_dict = dict(input_data)

        client = Traceloop.get()
        guardrails = Guardrails()

        eval_response = await guardrails.execute_evaluator(
            evaluator_slug=evaluator_slug,
            input=input_dict,
            async_http_client=client._async_http,
            evaluator_config=evaluator_config,
            input_schema=input_schema,
            timeout_in_sec=timeout_in_sec,
        )

        evaluator_result = eval_response.result

        # Record the evaluator result on the current span
        span = trace.get_current_span()
        if span is not None:
            try:
                span.set_attribute(GEN_AI_GUARDRAIL_OUTPUT, json.dumps(evaluator_result))
            except (TypeError, ValueError):
                span.set_attribute(GEN_AI_GUARDRAIL_OUTPUT, str(evaluator_result))

        if condition_field:
            result_to_validate = evaluator_result[condition_field]
        else:
            result_to_validate = evaluator_result

        return condition(result_to_validate)

    guard_fn.__name__ = evaluator_slug
    return guard_fn


def custom_evaluator_guard(
    evaluator_slug: str,
    evaluator_version: str | None = None,
    evaluator_config: dict[str, Any] | None = None,
    condition_field: str = "pass",
    condition: Callable[[Any], bool] = is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when the custom evaluator passes.

    Uses /v2/evaluators/{slug}/execute route.
    """

    async def guard_fn(input_data: Any) -> bool:
        from traceloop.sdk import Traceloop
        from traceloop.sdk.evaluator.evaluator import Evaluator

        if isinstance(input_data, dict):
            input_dict = input_data
        elif hasattr(input_data, "model_dump"):
            input_dict = input_data.model_dump()
        else:
            input_dict = dict(input_data)

        client = Traceloop.get()
        evaluator = Evaluator(async_http_client=client._async_http)

        eval_response = await evaluator.execute(
            evaluator_slug=evaluator_slug,
            input=input_dict,
            evaluator_version=evaluator_version,
            evaluator_config=evaluator_config,
            timeout_in_sec=timeout_in_sec,
        )

        evaluator_result = eval_response.result.evaluator_result

        span = trace.get_current_span()
        if span is not None:
            try:
                span.set_attribute(GEN_AI_GUARDRAIL_OUTPUT, json.dumps(evaluator_result))
            except (TypeError, ValueError):
                span.set_attribute(GEN_AI_GUARDRAIL_OUTPUT, str(evaluator_result))

        if condition_field:
            result_to_validate = evaluator_result[condition_field]
        else:
            result_to_validate = evaluator_result

        return condition(result_to_validate)

    guard_fn.__name__ = evaluator_slug
    return guard_fn


# =============================================================================
# Safety Detectors - pass when content is safe (is_safe = True)
# =============================================================================


def toxicity_guard(
    threshold: float | None = None,
    condition: Callable[[Any], bool] = is_true(),
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
    condition: Callable[[Any], bool] = is_true(),
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
    condition: Callable[[Any], bool] = is_true(),
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
    condition: Callable[[Any], bool] = is_false(),
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
    condition: Callable[[Any], bool] = is_false(),
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
    condition: Callable[[Any], bool] = is_false(),
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
# Validators - pass when valid (is_valid_* = True)
# =============================================================================


def json_validator_guard(
    enable_schema_validation: bool | None = None,
    schema_string: str | None = None,
    condition: Callable[[Any], bool] = is_true(),
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
    condition: Callable[[Any], bool] = is_true(),
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
    condition: Callable[[Any], bool] = is_true(),
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


# =============================================================================
# Quality and Adherence - pass when quality is good
# =============================================================================


def instruction_adherence_guard(
    condition: Callable[[Any], bool] = is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when instruction adherence score meets threshold."""
    return _create_guard(
        EvaluatorMadeByTraceloop.instruction_adherence(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def semantic_similarity_guard(
    condition: Callable[[Any], bool] = is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when semantic similarity score meets threshold."""
    return _create_guard(
        EvaluatorMadeByTraceloop.semantic_similarity(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def prompt_perplexity_guard(
    condition: Callable[[Any], bool] = is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when prompt perplexity score is below threshold."""
    return _create_guard(
        EvaluatorMadeByTraceloop.prompt_perplexity(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def uncertainty_guard(
    condition: Callable[[Any], bool] = is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when uncertainty is below threshold."""
    return _create_guard(
        EvaluatorMadeByTraceloop.uncertainty_detector(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )


def tone_detection_guard(
    condition: Callable[[Any], bool] = is_true(),
    timeout_in_sec: int = 60,
) -> Guard:
    """Guard that passes when detected tone matches expected."""
    return _create_guard(
        EvaluatorMadeByTraceloop.tone_detection(),
        condition=condition,
        timeout_in_sec=timeout_in_sec,
    )
