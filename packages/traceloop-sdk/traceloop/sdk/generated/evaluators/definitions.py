"""
Factory methods for creating Traceloop evaluators.

Provides type-safe factory methods with IDE autocomplete support.

DO NOT EDIT MANUALLY - Regenerate with:
    ./scripts/generate-models.sh /path/to/swagger.json
"""
from __future__ import annotations

from ...evaluator.config import EvaluatorDetails


class EvaluatorMadeByTraceloop:
    """
    Factory class for creating Traceloop evaluators with type-safe configuration.

    Each method creates an EvaluatorDetails instance for a specific evaluator,
    with properly typed configuration parameters.

    Example:
        >>> from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop
        >>>
        >>> evaluators = [
        ...     EvaluatorMadeByTraceloop.pii_detector(probability_threshold=0.8),
        ...     EvaluatorMadeByTraceloop.toxicity_detector(threshold=0.7),
        ...     EvaluatorMadeByTraceloop.faithfulness(),
        ... ]
    """

    @staticmethod
    def agent_efficiency() -> EvaluatorDetails:
        """Create agent-efficiency evaluator.

        Required input fields: trajectory_completions, trajectory_prompts
        """
        return EvaluatorDetails(
            slug="agent-efficiency",
            required_input_fields=['trajectory_completions', 'trajectory_prompts'],
        )

    @staticmethod
    def agent_flow_quality(
        conditions: list[str],
        threshold: float,
    ) -> EvaluatorDetails:
        """Create agent-flow-quality evaluator.

        Args:
            conditions: list[str]
            threshold: float

        Required input fields: trajectory_completions, trajectory_prompts
        """
        config = {
            k: v for k, v in {"conditions": conditions, "threshold": threshold}.items()
            if v is not None
        }
        return EvaluatorDetails(
            slug="agent-flow-quality",
            config=config if config else None,
            required_input_fields=['trajectory_completions', 'trajectory_prompts'],
        )

    @staticmethod
    def agent_goal_accuracy() -> EvaluatorDetails:
        """Create agent-goal-accuracy evaluator.

        Required input fields: completion, question, reference
        """
        return EvaluatorDetails(
            slug="agent-goal-accuracy",
            required_input_fields=['completion', 'question', 'reference'],
        )

    @staticmethod
    def agent_goal_completeness(
        threshold: float,
    ) -> EvaluatorDetails:
        """Create agent-goal-completeness evaluator.

        Args:
            threshold: float

        Required input fields: trajectory_completions, trajectory_prompts
        """
        config = {
            k: v for k, v in {"threshold": threshold}.items()
            if v is not None
        }
        return EvaluatorDetails(
            slug="agent-goal-completeness",
            config=config if config else None,
            required_input_fields=['trajectory_completions', 'trajectory_prompts'],
        )

    @staticmethod
    def agent_tool_error_detector() -> EvaluatorDetails:
        """Create agent-tool-error-detector evaluator.

        Required input fields: tool_input, tool_output
        """
        return EvaluatorDetails(
            slug="agent-tool-error-detector",
            required_input_fields=['tool_input', 'tool_output'],
        )

    @staticmethod
    def agent_tool_trajectory(
        input_params_sensitive: bool | None = None,
        mismatch_sensitive: bool | None = None,
        order_sensitive: bool | None = None,
        threshold: float | None = None,
    ) -> EvaluatorDetails:
        """Create agent-tool-trajectory evaluator.

        Args:
            input_params_sensitive: bool
            mismatch_sensitive: bool
            order_sensitive: bool
            threshold: float

        Required input fields: executed_tool_calls, expected_tool_calls
        """
        config = {
            k: v for k, v in {
                "input_params_sensitive": input_params_sensitive,
                "mismatch_sensitive": mismatch_sensitive,
                "order_sensitive": order_sensitive,
                "threshold": threshold,
            }.items()
            if v is not None
        }
        return EvaluatorDetails(
            slug="agent-tool-trajectory",
            config=config if config else None,
            required_input_fields=['executed_tool_calls', 'expected_tool_calls'],
        )

    @staticmethod
    def answer_completeness() -> EvaluatorDetails:
        """Create answer-completeness evaluator.

        Required input fields: completion, context, question
        """
        return EvaluatorDetails(
            slug="answer-completeness",
            required_input_fields=['completion', 'context', 'question'],
        )

    @staticmethod
    def answer_correctness() -> EvaluatorDetails:
        """Create answer-correctness evaluator.

        Required input fields: completion, ground_truth, question
        """
        return EvaluatorDetails(
            slug="answer-correctness",
            required_input_fields=['completion', 'ground_truth', 'question'],
        )

    @staticmethod
    def answer_relevancy() -> EvaluatorDetails:
        """Create answer-relevancy evaluator.

        Required input fields: answer, question
        """
        return EvaluatorDetails(
            slug="answer-relevancy",
            required_input_fields=['answer', 'question'],
        )

    @staticmethod
    def char_count() -> EvaluatorDetails:
        """Create char-count evaluator.

        Required input fields: text
        """
        return EvaluatorDetails(
            slug="char-count",
            required_input_fields=['text'],
        )

    @staticmethod
    def char_count_ratio() -> EvaluatorDetails:
        """Create char-count-ratio evaluator.

        Required input fields: denominator_text, numerator_text
        """
        return EvaluatorDetails(
            slug="char-count-ratio",
            required_input_fields=['denominator_text', 'numerator_text'],
        )

    @staticmethod
    def context_relevance(
        model: str | None = None,
    ) -> EvaluatorDetails:
        """Create context-relevance evaluator.

        Args:
            model: str

        Required input fields: context, query
        """
        config = {
            k: v for k, v in {"model": model}.items()
            if v is not None
        }
        return EvaluatorDetails(
            slug="context-relevance",
            config=config if config else None,
            required_input_fields=['context', 'query'],
        )

    @staticmethod
    def conversation_quality() -> EvaluatorDetails:
        """Create conversation-quality evaluator.

        Required input fields: completions, prompts
        """
        return EvaluatorDetails(
            slug="conversation-quality",
            required_input_fields=['completions', 'prompts'],
        )

    @staticmethod
    def faithfulness() -> EvaluatorDetails:
        """Create faithfulness evaluator.

        Required input fields: completion, context, question
        """
        return EvaluatorDetails(
            slug="faithfulness",
            required_input_fields=['completion', 'context', 'question'],
        )

    @staticmethod
    def html_comparison() -> EvaluatorDetails:
        """Create html-comparison evaluator.

        Required input fields: html1, html2
        """
        return EvaluatorDetails(
            slug="html-comparison",
            required_input_fields=['html1', 'html2'],
        )

    @staticmethod
    def instruction_adherence() -> EvaluatorDetails:
        """Create instruction-adherence evaluator.

        Required input fields: instructions, response
        """
        return EvaluatorDetails(
            slug="instruction-adherence",
            required_input_fields=['instructions', 'response'],
        )

    @staticmethod
    def intent_change() -> EvaluatorDetails:
        """Create intent-change evaluator.

        Required input fields: completions, prompts
        """
        return EvaluatorDetails(
            slug="intent-change",
            required_input_fields=['completions', 'prompts'],
        )

    @staticmethod
    def json_validator(
        enable_schema_validation: bool | None = None,
        schema_string: str | None = None,
    ) -> EvaluatorDetails:
        """Create json-validator evaluator.

        Args:
            enable_schema_validation: bool
            schema_string: str

        Required input fields: text
        """
        config = {
            k: v for k, v in {
                "enable_schema_validation": enable_schema_validation,
                "schema_string": schema_string,
            }.items()
            if v is not None
        }
        return EvaluatorDetails(
            slug="json-validator",
            config=config if config else None,
            required_input_fields=['text'],
        )

    @staticmethod
    def perplexity() -> EvaluatorDetails:
        """Create perplexity evaluator.

        Required input fields: logprobs
        """
        return EvaluatorDetails(
            slug="perplexity",
            required_input_fields=['logprobs'],
        )

    @staticmethod
    def pii_detector(
        probability_threshold: float | None = None,
    ) -> EvaluatorDetails:
        """Create pii-detector evaluator.

        Args:
            probability_threshold: float

        Required input fields: text
        """
        config = {
            k: v for k, v in {"probability_threshold": probability_threshold}.items()
            if v is not None
        }
        return EvaluatorDetails(
            slug="pii-detector",
            config=config if config else None,
            required_input_fields=['text'],
        )

    @staticmethod
    def placeholder_regex(
        case_sensitive: bool | None = None,
        dot_include_nl: bool | None = None,
        multi_line: bool | None = None,
        should_match: bool | None = None,
    ) -> EvaluatorDetails:
        """Create placeholder-regex evaluator.

        Args:
            case_sensitive: bool
            dot_include_nl: bool
            multi_line: bool
            should_match: bool

        Required input fields: placeholder_value, text
        """
        config = {
            k: v for k, v in {
                "case_sensitive": case_sensitive,
                "dot_include_nl": dot_include_nl,
                "multi_line": multi_line,
                "should_match": should_match,
            }.items()
            if v is not None
        }
        return EvaluatorDetails(
            slug="placeholder-regex",
            config=config if config else None,
            required_input_fields=['placeholder_value', 'text'],
        )

    @staticmethod
    def profanity_detector() -> EvaluatorDetails:
        """Create profanity-detector evaluator.

        Required input fields: text
        """
        return EvaluatorDetails(
            slug="profanity-detector",
            required_input_fields=['text'],
        )

    @staticmethod
    def prompt_injection(
        threshold: float | None = None,
    ) -> EvaluatorDetails:
        """Create prompt-injection evaluator.

        Args:
            threshold: float

        Required input fields: prompt
        """
        config = {
            k: v for k, v in {"threshold": threshold}.items()
            if v is not None
        }
        return EvaluatorDetails(
            slug="prompt-injection",
            config=config if config else None,
            required_input_fields=['prompt'],
        )

    @staticmethod
    def prompt_perplexity() -> EvaluatorDetails:
        """Create prompt-perplexity evaluator.

        Required input fields: prompt
        """
        return EvaluatorDetails(
            slug="prompt-perplexity",
            required_input_fields=['prompt'],
        )

    @staticmethod
    def regex_validator(
        case_sensitive: bool | None = None,
        dot_include_nl: bool | None = None,
        multi_line: bool | None = None,
        regex: str | None = None,
        should_match: bool | None = None,
    ) -> EvaluatorDetails:
        """Create regex-validator evaluator.

        Args:
            case_sensitive: bool
            dot_include_nl: bool
            multi_line: bool
            regex: str
            should_match: bool

        Required input fields: text
        """
        config = {
            k: v for k, v in {
                "case_sensitive": case_sensitive,
                "dot_include_nl": dot_include_nl,
                "multi_line": multi_line,
                "regex": regex,
                "should_match": should_match,
            }.items()
            if v is not None
        }
        return EvaluatorDetails(
            slug="regex-validator",
            config=config if config else None,
            required_input_fields=['text'],
        )

    @staticmethod
    def secrets_detector() -> EvaluatorDetails:
        """Create secrets-detector evaluator.

        Required input fields: text
        """
        return EvaluatorDetails(
            slug="secrets-detector",
            required_input_fields=['text'],
        )

    @staticmethod
    def semantic_similarity() -> EvaluatorDetails:
        """Create semantic-similarity evaluator.

        Required input fields: completion, reference
        """
        return EvaluatorDetails(
            slug="semantic-similarity",
            required_input_fields=['completion', 'reference'],
        )

    @staticmethod
    def sexism_detector(
        threshold: float | None = None,
    ) -> EvaluatorDetails:
        """Create sexism-detector evaluator.

        Args:
            threshold: float

        Required input fields: text
        """
        config = {
            k: v for k, v in {"threshold": threshold}.items()
            if v is not None
        }
        return EvaluatorDetails(
            slug="sexism-detector",
            config=config if config else None,
            required_input_fields=['text'],
        )

    @staticmethod
    def sql_validator() -> EvaluatorDetails:
        """Create sql-validator evaluator.

        Required input fields: text
        """
        return EvaluatorDetails(
            slug="sql-validator",
            required_input_fields=['text'],
        )

    @staticmethod
    def tone_detection() -> EvaluatorDetails:
        """Create tone-detection evaluator.

        Required input fields: text
        """
        return EvaluatorDetails(
            slug="tone-detection",
            required_input_fields=['text'],
        )

    @staticmethod
    def topic_adherence() -> EvaluatorDetails:
        """Create topic-adherence evaluator.

        Required input fields: completion, question, reference_topics
        """
        return EvaluatorDetails(
            slug="topic-adherence",
            required_input_fields=['completion', 'question', 'reference_topics'],
        )

    @staticmethod
    def toxicity_detector(
        threshold: float | None = None,
    ) -> EvaluatorDetails:
        """Create toxicity-detector evaluator.

        Args:
            threshold: float

        Required input fields: text
        """
        config = {
            k: v for k, v in {"threshold": threshold}.items()
            if v is not None
        }
        return EvaluatorDetails(
            slug="toxicity-detector",
            config=config if config else None,
            required_input_fields=['text'],
        )

    @staticmethod
    def uncertainty_detector() -> EvaluatorDetails:
        """Create uncertainty-detector evaluator.

        Required input fields: prompt
        """
        return EvaluatorDetails(
            slug="uncertainty-detector",
            required_input_fields=['prompt'],
        )

    @staticmethod
    def word_count() -> EvaluatorDetails:
        """Create word-count evaluator.

        Required input fields: text
        """
        return EvaluatorDetails(
            slug="word-count",
            required_input_fields=['text'],
        )

    @staticmethod
    def word_count_ratio() -> EvaluatorDetails:
        """Create word-count-ratio evaluator.

        Required input fields: denominator_text, numerator_text
        """
        return EvaluatorDetails(
            slug="word-count-ratio",
            required_input_fields=['denominator_text', 'numerator_text'],
        )
