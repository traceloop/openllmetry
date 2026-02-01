"""
Guardrail module for the Traceloop SDK.

Provides a simple function-based guardrail system for running protected operations
with evaluation and failure handling.

Example:
    from traceloop.sdk import Traceloop
    from traceloop.sdk.guardrail import pii_guard, OnFailure

    # Initialize and get client
    client = Traceloop.init(api_key="...")

    async def generate_email() -> str:
        return await llm.complete("Write a customer email...")

    guardrail = client.guardrails.create(
        guards=[pii_guard()],
        on_failure=OnFailure.raise_exception("PII detected in response"),
    )
    result = await guardrail.run(generate_email)

    # With custom input mapper
    result = await guardrail.run(
        generate_email,
        input_mapper=lambda text: [{"text": text}]
    )
"""

from .guardrail import Guardrails
from .model import (
    GuardedResult,
    GuardrailError,
    GuardValidationError,
    GuardExecutionError,
    GuardInputTypeError,
    Guard,
    OnFailureHandler,
    InputMapper,
    GuardInput,
    GuardedFunctionResult,
)
from .condition import Condition
from .on_failure import OnFailure
from .guards import (
    custom_evaluator_guard,
    toxicity_guard,
    profanity_guard,
    sexism_guard,
    pii_guard,
    secrets_guard,
    prompt_injection_guard,
    answer_relevancy_guard,
    faithfulness_guard,
    json_validator_guard,
    sql_validator_guard,
    regex_validator_guard,
    placeholder_regex_guard,
    agent_efficiency_guard,
    agent_flow_quality_guard,
    agent_goal_accuracy_guard,
    agent_goal_completeness_guard,
    agent_tool_error_guard,
    agent_tool_trajectory_guard,
    intent_change_guard,
    answer_correctness_guard,
    answer_completeness_guard,
    context_relevance_guard,
    conversation_quality_guard,
    html_comparison_guard,
    instruction_adherence_guard,
    semantic_similarity_guard,
    topic_adherence_guard,
    perplexity_guard,
    prompt_perplexity_guard,
    uncertainty_guard,
    word_count_guard,
    char_count_guard,
    word_count_ratio_guard,
    char_count_ratio_guard,
    tone_detection_guard,
)
from .default_mapper import default_input_mapper

__all__ = [
    "Guardrails",
    "GuardedResult",
    "GuardrailError",
    "GuardValidationError",
    "GuardExecutionError",
    "GuardInputTypeError",
    "Guard",
    "OnFailureHandler",
    "InputMapper",
    "GuardInput",
    "GuardedFunctionResult",
    "Condition",
    "OnFailure",
    "default_input_mapper",
    # Guard functions
    "custom_evaluator_guard",
    "toxicity_guard",
    "profanity_guard",
    "sexism_guard",
    "pii_guard",
    "secrets_guard",
    "prompt_injection_guard",
    "answer_relevancy_guard",
    "faithfulness_guard",
    "json_validator_guard",
    "sql_validator_guard",
    "regex_validator_guard",
    "placeholder_regex_guard",
    "agent_efficiency_guard",
    "agent_flow_quality_guard",
    "agent_goal_accuracy_guard",
    "agent_goal_completeness_guard",
    "agent_tool_error_guard",
    "agent_tool_trajectory_guard",
    "intent_change_guard",
    "answer_correctness_guard",
    "answer_completeness_guard",
    "context_relevance_guard",
    "conversation_quality_guard",
    "html_comparison_guard",
    "instruction_adherence_guard",
    "semantic_similarity_guard",
    "topic_adherence_guard",
    "perplexity_guard",
    "prompt_perplexity_guard",
    "uncertainty_guard",
    "word_count_guard",
    "char_count_guard",
    "word_count_ratio_guard",
    "char_count_ratio_guard",
    "tone_detection_guard",
]
