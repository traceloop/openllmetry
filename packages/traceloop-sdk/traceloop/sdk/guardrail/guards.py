"""
Simplified guard factory with sensible defaults.

Provides pre-configured guards ready for use with client.guardrails.create().

Example:
    from traceloop.sdk import Traceloop
    from traceloop.sdk.guardrail import Guards, OnFailure

    client = Traceloop.init(api_key="...")

    guardrail = client.guardrails.create(
        guards=[
            Guards.toxicity_detector(),
            Guards.pii_detector(),
            Guards.answer_relevancy(),
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


def guard(
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


class Guards:
    """
    Simplified guardrail factory with sensible defaults.

    Each method returns an async guard function ready for use with
    client.guardrails.create(). Default conditions are automatically
    applied based on the evaluator type.

    Categories:
    - Safety detectors (toxicity, profanity, sexism): pass when is_safe == True
    - Detection guards (pii, secrets, prompt_injection): pass when has_* == False
    - Quality evaluators (answer_relevancy, faithfulness): pass when is_* == True
    - Score-based evaluators: pass when score >= threshold (default 0.8)
    """

    @staticmethod
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
        return guard(evaluator_details, condition, timeout_in_sec)

    # =========================================================================
    # Safety Detectors - pass when content is safe (is_safe = True)
    # =========================================================================

    @staticmethod
    def toxicity_detector(
        threshold: float | None = None,
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when content is safe from toxicity."""
        return guard(
            EvaluatorMadeByTraceloop.toxicity_detector(threshold=threshold),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def profanity_detector(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when content is free from profanity."""
        return guard(
            EvaluatorMadeByTraceloop.profanity_detector(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def sexism_detector(
        threshold: float | None = None,
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when content is free from sexism."""
        return guard(
            EvaluatorMadeByTraceloop.sexism_detector(threshold=threshold),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    # =========================================================================
    # Detection Guards - pass when NOT detected (has_* = False)
    # =========================================================================

    @staticmethod
    def pii_detector(
        probability_threshold: float | None = None,
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when no PII is detected."""
        return guard(
            EvaluatorMadeByTraceloop.pii_detector(
                probability_threshold=probability_threshold
            ),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def secrets_detector(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when no secrets are detected."""
        return guard(
            EvaluatorMadeByTraceloop.secrets_detector(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def prompt_injection(
        threshold: float | None = None,
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when no prompt injection is detected."""
        return guard(
            EvaluatorMadeByTraceloop.prompt_injection(threshold=threshold),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    # =========================================================================
    # Quality Evaluators - pass when quality is good (is_* = True)
    # =========================================================================

    @staticmethod
    def answer_relevancy(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when the answer is relevant to the question."""
        return guard(
            EvaluatorMadeByTraceloop.answer_relevancy(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def faithfulness(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when the response is faithful to the context."""
        return guard(
            EvaluatorMadeByTraceloop.faithfulness(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    # =========================================================================
    # Validators - pass when valid (is_valid_* = True)
    # =========================================================================

    @staticmethod
    def json_validator(
        enable_schema_validation: bool | None = None,
        schema_string: str | None = None,
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when the text is valid JSON."""
        return guard(
            EvaluatorMadeByTraceloop.json_validator(
                enable_schema_validation=enable_schema_validation,
                schema_string=schema_string,
            ),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def sql_validator(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when the text is valid SQL."""
        return guard(
            EvaluatorMadeByTraceloop.sql_validator(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def regex_validator(
        regex: str | None = None,
        case_sensitive: bool | None = None,
        dot_include_nl: bool | None = None,
        multi_line: bool | None = None,
        should_match: bool | None = None,
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when the text matches the regex pattern."""
        return guard(
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

    @staticmethod
    def placeholder_regex(
        case_sensitive: bool | None = None,
        dot_include_nl: bool | None = None,
        multi_line: bool | None = None,
        should_match: bool | None = None,
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when placeholder regex validation succeeds."""
        return guard(
            EvaluatorMadeByTraceloop.placeholder_regex(
                case_sensitive=case_sensitive,
                dot_include_nl=dot_include_nl,
                multi_line=multi_line,
                should_match=should_match,
            ),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    # =========================================================================
    # Agent Evaluators - pass on success
    # =========================================================================

    @staticmethod
    def agent_efficiency(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when agent efficiency score meets threshold."""
        return guard(
            EvaluatorMadeByTraceloop.agent_efficiency(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def agent_flow_quality(
        conditions: list[str],
        threshold: float,
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when agent flow meets quality conditions."""
        return guard(
            EvaluatorMadeByTraceloop.agent_flow_quality(
                conditions=conditions,
                threshold=threshold,
            ),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def agent_goal_accuracy(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when agent goal accuracy score meets threshold."""
        return guard(
            EvaluatorMadeByTraceloop.agent_goal_accuracy(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def agent_goal_completeness(
        threshold: float,
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when agent completes its goal."""
        return guard(
            EvaluatorMadeByTraceloop.agent_goal_completeness(threshold=threshold),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def agent_tool_error_detector(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when no tool errors are detected."""
        return guard(
            EvaluatorMadeByTraceloop.agent_tool_error_detector(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def agent_tool_trajectory(
        input_params_sensitive: bool | None = None,
        mismatch_sensitive: bool | None = None,
        order_sensitive: bool | None = None,
        threshold: float | None = None,
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when agent tool trajectory matches expected."""
        return guard(
            EvaluatorMadeByTraceloop.agent_tool_trajectory(
                input_params_sensitive=input_params_sensitive,
                mismatch_sensitive=mismatch_sensitive,
                order_sensitive=order_sensitive,
                threshold=threshold,
            ),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def intent_change(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when intent remains consistent."""
        return guard(
            EvaluatorMadeByTraceloop.intent_change(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    # =========================================================================
    # Score-based Evaluators - pass when score >= threshold (default 0.8)
    # =========================================================================

    @staticmethod
    def answer_correctness(
        threshold: float = 0.8,
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when answer correctness score meets threshold."""
        return guard(
            EvaluatorMadeByTraceloop.answer_correctness(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def answer_completeness(
        threshold: float = 0.8,
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when answer completeness score meets threshold."""
        return guard(
            EvaluatorMadeByTraceloop.answer_completeness(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def context_relevance(
        threshold: float = 0.8,
        model: str | None = None,
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when context relevance score meets threshold."""
        return guard(
            EvaluatorMadeByTraceloop.context_relevance(model=model),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def conversation_quality(
        threshold: float = 0.7,
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when conversation quality score meets threshold."""
        return guard(
            EvaluatorMadeByTraceloop.conversation_quality(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def html_comparison(
        threshold: float = 0.9,
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when HTML similarity score meets threshold."""
        return guard(
            EvaluatorMadeByTraceloop.html_comparison(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def instruction_adherence(
        threshold: float = 0.8,
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when instruction adherence score meets threshold."""
        return guard(
            EvaluatorMadeByTraceloop.instruction_adherence(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def semantic_similarity(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when semantic similarity score meets threshold."""
        return guard(
            EvaluatorMadeByTraceloop.semantic_similarity(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def topic_adherence(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when topic adherence score meets threshold."""
        return guard(
            EvaluatorMadeByTraceloop.topic_adherence(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    # =========================================================================
    # Score-based Evaluators (lower is better) - pass when score <= threshold
    # =========================================================================

    @staticmethod
    def perplexity(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when perplexity score is below threshold."""
        return guard(
            EvaluatorMadeByTraceloop.perplexity(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def prompt_perplexity(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when prompt perplexity score is below threshold."""
        return guard(
            EvaluatorMadeByTraceloop.prompt_perplexity(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def uncertainty_detector(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when uncertainty is below threshold."""
        return guard(
            EvaluatorMadeByTraceloop.uncertainty_detector(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    # =========================================================================
    # Count/Ratio Evaluators - require explicit min/max bounds
    # =========================================================================

    @staticmethod
    def word_count(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when word count is within specified bounds."""
        return guard(
            EvaluatorMadeByTraceloop.word_count(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def char_count(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when character count is within specified bounds."""
        return guard(
            EvaluatorMadeByTraceloop.char_count(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def word_count_ratio(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when word count ratio is within specified bounds."""
        return guard(
            EvaluatorMadeByTraceloop.word_count_ratio(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    @staticmethod
    def char_count_ratio(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when character count ratio is within specified bounds."""
        return guard(
            EvaluatorMadeByTraceloop.char_count_ratio(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )

    # =========================================================================
    # Special Evaluators
    # =========================================================================

    @staticmethod
    def tone_detection(
        condition: Callable[[Any], bool] = Condition.is_true(),
        timeout_in_sec: int = 60,
    ) -> Guard:
        """Guard that passes when detected tone matches expected."""
        return guard(
            EvaluatorMadeByTraceloop.tone_detection(),
            condition=condition,
            timeout_in_sec=timeout_in_sec,
        )
