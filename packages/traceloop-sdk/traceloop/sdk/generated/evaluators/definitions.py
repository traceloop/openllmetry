"""
Factory methods for creating Traceloop evaluators.

Provides type-safe factory methods with IDE autocomplete support.

DO NOT EDIT MANUALLY - Regenerate with:
    ./scripts/generate-models.sh /path/to/swagger.json
"""
from __future__ import annotations

from ...evaluator.config import EvaluatorDetails
from traceloop.sdk.generated.evaluators.response import (
    AgentEfficiencyResponse,
    AgentFlowQualityResponse,
    AgentGoalAccuracyResponse,
    AgentGoalCompletenessResponse,
    AgentToolErrorDetectorResponse,
    AgentToolTrajectoryResponse,
    AnswerCompletenessResponse,
    AnswerCorrectnessResponse,
    AnswerRelevancyResponse,
    CharCountRatioResponse,
    CharCountResponse,
    ContextRelevanceResponse,
    ConversationQualityResponse,
    FaithfulnessResponse,
    HtmlComparisonResponse,
    InstructionAdherenceResponse,
    IntentChangeResponse,
    JSONValidatorResponse,
    PerplexityResponse,
    PIIDetectorResponse,
    PlaceholderRegexResponse,
    ProfanityDetectorResponse,
    PromptInjectionResponse,
    PromptPerplexityResponse,
    RegexValidatorResponse,
    SQLValidatorResponse,
    SecretsDetectorResponse,
    SemanticSimilarityResponse,
    SexismDetectorResponse,
    ToneDetectionResponse,
    TopicAdherenceResponse,
    ToxicityDetectorResponse,
    UncertaintyDetectorResponse,
    WordCountRatioResponse,
    WordCountResponse,
)


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
            condition_field="task_completion_score",
            output_schema=AgentEfficiencyResponse,
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
            condition_field="success",
            output_schema=AgentFlowQualityResponse,
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
            condition_field="accuracy_score",
            output_schema=AgentGoalAccuracyResponse,
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
            condition_field="success",
            output_schema=AgentGoalCompletenessResponse,
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
            condition_field="success",
            output_schema=AgentToolErrorDetectorResponse,
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
            condition_field="success",
            output_schema=AgentToolTrajectoryResponse,
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
            condition_field="answer_completeness_score",
            output_schema=AnswerCompletenessResponse,
            required_input_fields=['completion', 'context', 'question'],
        )

    @staticmethod
    def answer_correctness() -> EvaluatorDetails:
        """Create answer-correctness evaluator.

        Required input fields: completion, ground_truth, question
        """
        return EvaluatorDetails(
            slug="answer-correctness",
            condition_field="correctness_score",
            output_schema=AnswerCorrectnessResponse,
            required_input_fields=['completion', 'ground_truth', 'question'],
        )

    @staticmethod
    def answer_relevancy() -> EvaluatorDetails:
        """Create answer-relevancy evaluator.

        Required input fields: answer, question
        """
        return EvaluatorDetails(
            slug="answer-relevancy",
            condition_field="is_relevant",
            output_schema=AnswerRelevancyResponse,
            required_input_fields=['answer', 'question'],
        )

    @staticmethod
    def char_count() -> EvaluatorDetails:
        """Create char-count evaluator.

        Required input fields: text
        """
        return EvaluatorDetails(
            slug="char-count",
            condition_field="char_count",
            output_schema=CharCountResponse,
            required_input_fields=['text'],
        )

    @staticmethod
    def char_count_ratio() -> EvaluatorDetails:
        """Create char-count-ratio evaluator.

        Required input fields: denominator_text, numerator_text
        """
        return EvaluatorDetails(
            slug="char-count-ratio",
            condition_field="char_ratio",
            output_schema=CharCountRatioResponse,
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
            condition_field="relevance_score",
            output_schema=ContextRelevanceResponse,
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
            condition_field="conversation_quality_score",
            output_schema=ConversationQualityResponse,
            required_input_fields=['completions', 'prompts'],
        )

    @staticmethod
    def faithfulness() -> EvaluatorDetails:
        """Create faithfulness evaluator.

        Required input fields: completion, context, question
        """
        return EvaluatorDetails(
            slug="faithfulness",
            condition_field="is_faithful",
            output_schema=FaithfulnessResponse,
            required_input_fields=['completion', 'context', 'question'],
        )

    @staticmethod
    def html_comparison() -> EvaluatorDetails:
        """Create html-comparison evaluator.

        Required input fields: html1, html2
        """
        return EvaluatorDetails(
            slug="html-comparison",
            condition_field="similarity_score",
            output_schema=HtmlComparisonResponse,
            required_input_fields=['html1', 'html2'],
        )

    @staticmethod
    def instruction_adherence() -> EvaluatorDetails:
        """Create instruction-adherence evaluator.

        Required input fields: instructions, response
        """
        return EvaluatorDetails(
            slug="instruction-adherence",
            condition_field="instruction_adherence_score",
            output_schema=InstructionAdherenceResponse,
            required_input_fields=['instructions', 'response'],
        )

    @staticmethod
    def intent_change() -> EvaluatorDetails:
        """Create intent-change evaluator.

        Required input fields: completions, prompts
        """
        return EvaluatorDetails(
            slug="intent-change",
            condition_field="success",
            output_schema=IntentChangeResponse,
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
            condition_field="is_valid_json",
            output_schema=JSONValidatorResponse,
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
            condition_field="perplexity_score",
            output_schema=PerplexityResponse,
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
            condition_field="has_pii",
            output_schema=PIIDetectorResponse,
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
            condition_field="is_valid_regex",
            output_schema=PlaceholderRegexResponse,
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
            condition_field="is_safe",
            output_schema=ProfanityDetectorResponse,
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
            condition_field="has_injection",
            output_schema=PromptInjectionResponse,
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
            condition_field="perplexity_score",
            output_schema=PromptPerplexityResponse,
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
            condition_field="is_valid_regex",
            output_schema=RegexValidatorResponse,
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
            condition_field="has_secret",
            output_schema=SecretsDetectorResponse,
            required_input_fields=['text'],
        )

    @staticmethod
    def semantic_similarity() -> EvaluatorDetails:
        """Create semantic-similarity evaluator.

        Required input fields: completion, reference
        """
        return EvaluatorDetails(
            slug="semantic-similarity",
            condition_field="similarity_score",
            output_schema=SemanticSimilarityResponse,
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
            condition_field="is_safe",
            output_schema=SexismDetectorResponse,
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
            condition_field="is_valid_sql",
            output_schema=SQLValidatorResponse,
            required_input_fields=['text'],
        )

    @staticmethod
    def tone_detection() -> EvaluatorDetails:
        """Create tone-detection evaluator.

        Required input fields: text
        """
        return EvaluatorDetails(
            slug="tone-detection",
            condition_field="tone",
            output_schema=ToneDetectionResponse,
            required_input_fields=['text'],
        )

    @staticmethod
    def topic_adherence() -> EvaluatorDetails:
        """Create topic-adherence evaluator.

        Required input fields: completion, question, reference_topics
        """
        return EvaluatorDetails(
            slug="topic-adherence",
            condition_field="adherence_score",
            output_schema=TopicAdherenceResponse,
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
            condition_field="is_safe",
            output_schema=ToxicityDetectorResponse,
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
            condition_field="uncertainty",
            output_schema=UncertaintyDetectorResponse,
            required_input_fields=['prompt'],
        )

    @staticmethod
    def word_count() -> EvaluatorDetails:
        """Create word-count evaluator.

        Required input fields: text
        """
        return EvaluatorDetails(
            slug="word-count",
            condition_field="word_count",
            output_schema=WordCountResponse,
            required_input_fields=['text'],
        )

    @staticmethod
    def word_count_ratio() -> EvaluatorDetails:
        """Create word-count-ratio evaluator.

        Required input fields: denominator_text, numerator_text
        """
        return EvaluatorDetails(
            slug="word-count-ratio",
            condition_field="word_ratio",
            output_schema=WordCountRatioResponse,
            required_input_fields=['denominator_text', 'numerator_text'],
        )
