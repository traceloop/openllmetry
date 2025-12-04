from typing import Optional, Dict, Any
from .config import EvaluatorDetails


class EvaluatorMadeByTraceloop:
    """
    Factory class for creating made by traceloop evaluators with proper configuration.

    This class provides easy-to-use factory methods for all made by traceloop evaluators,
    with type hints and documentation for their configuration options.

    Example:
        >>> from traceloop.sdk.evaluator import Predefined
        >>>
        >>> evaluators = [
        ...     EvaluatorMadeByTraceloop.pii_detector(probability_threshold=0.8),
        ...     EvaluatorMadeByTraceloop.toxicity_detector(threshold=0.7),
        ... ]
    """

    @staticmethod
    def pii_detector(
        probability_threshold: float = 0.5,
    ) -> EvaluatorDetails:
        """
        PII (Personally Identifiable Information) detector evaluator.

        Args:
            probability_threshold: Minimum probability threshold for detecting PII (0.0-1.0)

        Returns:
            EvaluatorDetails configured for PII detection
        """
        config: Dict[str, Any] = {"probability_threshold": probability_threshold}
        return EvaluatorDetails(slug="pii-detector", version=None, config=config)

    @staticmethod
    def toxicity_detector(
        threshold: float = 0.5,
    ) -> EvaluatorDetails:
        """
        Toxicity detector evaluator.

        Args:
            threshold: Minimum toxicity threshold for flagging content (0.0-1.0)

        Returns:
            EvaluatorDetails configured for toxicity detection
        """
        config: Dict[str, Any] = {"threshold": threshold}

        return EvaluatorDetails(slug="toxicity-detector", version=None, config=config)

    @staticmethod
    def prompt_injection(
        threshold: float = 0.5,
    ) -> EvaluatorDetails:
        """
        Prompt injection detector evaluator.

        Args:
            threshold: Minimum threshold for detecting prompt injection attempts (0.0-1.0)

        Returns:
            EvaluatorDetails configured for prompt injection detection
        """
        config: Dict[str, Any] = {"threshold": threshold}
        return EvaluatorDetails(slug="prompt-injection", version=None, config=config)

    @staticmethod
    def regex_validator(
        regex: str,
        should_match: bool = True,
        case_sensitive: bool = True,
        dot_include_nl: bool = False,
        multi_line: bool = False,
    ) -> EvaluatorDetails:
        """
        Regular expression validator evaluator.

        Args:
            regex: The regular expression pattern to match against
            should_match: If True, pass when pattern matches; if False, pass when pattern doesn't match
            case_sensitive: Whether the regex matching should be case-sensitive
            dot_include_nl: Whether the dot (.) should match newline characters
            multi_line: Whether to enable multi-line mode (^ and $ match line boundaries)

        Returns:
            EvaluatorDetails configured for regex validation
        """
        config: Dict[str, Any] = {
            "regex": regex,
            "should_match": should_match,
            "case_sensitive": case_sensitive,
            "dot_include_nl": dot_include_nl,
            "multi_line": multi_line,
        }

        return EvaluatorDetails(slug="regex-validator", version=None, config=config)

    @staticmethod
    def json_validator(
        enable_schema_validation: bool = False,
        schema_string: Optional[str] = None,
    ) -> EvaluatorDetails:
        """
        JSON validator evaluator.

        Args:
            enable_schema_validation: Whether to validate against a JSON schema
            schema_string: JSON schema string to validate against (required if enable_schema_validation is True)

        Returns:
            EvaluatorDetails configured for JSON validation
        """
        config: Dict[str, Any] = {
            "enable_schema_validation": enable_schema_validation,
        }
        if schema_string:
            config["schema_string"] = schema_string

        return EvaluatorDetails(slug="json-validator", version=None, config=config)

    @staticmethod
    def placeholder_regex(
        regex: str,
        placeholder_name: str,
        should_match: bool = True,
        case_sensitive: bool = True,
        dot_include_nl: bool = False,
        multi_line: bool = False,
    ) -> EvaluatorDetails:
        """
        Placeholder regex evaluator - validates that placeholders match a regex pattern.

        Args:
            regex: The regular expression pattern to match against
            placeholder_name: Name of the placeholder to validate
            should_match: If True, pass when pattern matches; if False, pass when pattern doesn't match
            case_sensitive: Whether the regex matching should be case-sensitive
            dot_include_nl: Whether the dot (.) should match newline characters
            multi_line: Whether to enable multi-line mode (^ and $ match line boundaries)

        Returns:
            EvaluatorDetails configured for placeholder regex validation
        """
        config: Dict[str, Any] = {
            "regex": regex,
            "placeholder_name": placeholder_name,
            "should_match": should_match,
            "case_sensitive": case_sensitive,
            "dot_include_nl": dot_include_nl,
            "multi_line": multi_line,
        }

        return EvaluatorDetails(slug="placeholder-regex", version=None, config=config)

    @staticmethod
    def char_count(
    ) -> EvaluatorDetails:
        """
        Character count evaluator - counts the number of characters in text.

        Returns:
            EvaluatorDetails configured for character counting
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(slug="char-count", version=None, config=config)

    @staticmethod
    def char_count_ratio(
    ) -> EvaluatorDetails:
        """
        Character count ratio evaluator - measures the ratio of characters between two texts.

        Returns:
            EvaluatorDetails configured for character count ratio calculation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(slug="char-count-ratio", version=None, config=config)

    @staticmethod
    def word_count() -> EvaluatorDetails:
        """
        Word count evaluator - counts the number of words in text.

        Returns:
            EvaluatorDetails configured for word counting
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(slug="word-count", version=None, config=config)

    @staticmethod
    def word_count_ratio(
    ) -> EvaluatorDetails:
        """
        Word count ratio evaluator - measures the ratio of words between two texts.

        Returns:
            EvaluatorDetails configured for word count ratio calculation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(slug="word-count-ratio", version=None, config=config)

    @staticmethod
    def answer_relevancy(
    ) -> EvaluatorDetails:
        """
        Answer relevancy evaluator - verifies responses address the query.

        Returns:
            EvaluatorDetails configured for answer relevancy evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(slug="answer-relevancy", version=None, config=config)

    @staticmethod
    def faithfulness(
    ) -> EvaluatorDetails:
        """
        Faithfulness evaluator - detects hallucinations and verifies facts.

        Returns:
            EvaluatorDetails configured for faithfulness evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(slug="faithfulness", version=None, config=config)

    @staticmethod
    def profanity_detector() -> EvaluatorDetails:
        """
        Profanity detector evaluator - flags inappropriate language.

        Returns:
            EvaluatorDetails configured for profanity detection
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(slug="profanity-detector", version=None, config=config)

    @staticmethod
    def sexism_detector(
        threshold: float = 0.5,
    ) -> EvaluatorDetails:
        """
        Sexism detector evaluator - detects sexist language and bias.

        Args:
            threshold: Minimum threshold for detecting sexism (0.0-1.0)

        Returns:
            EvaluatorDetails configured for sexism detection
        """
        config: Dict[str, Any] = {"threshold": threshold}

        return EvaluatorDetails(slug="sexism-detector", version=None, config=config)

    @staticmethod
    def secrets_detector(
    ) -> EvaluatorDetails:
        """
        Secrets detector evaluator - monitors for credential and key leaks.

        Returns:
            EvaluatorDetails configured for secrets detection
        """
        return EvaluatorDetails(slug="secrets-detector", version=None, config=None)

    @staticmethod
    def sql_validator(
    ) -> EvaluatorDetails:
        """
        SQL validator evaluator - validates SQL queries.

        Returns:
            EvaluatorDetails configured for SQL validation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(slug="sql-validator", version=None, config=config)

    @staticmethod
    def semantic_similarity(
    ) -> EvaluatorDetails:
        """
        Semantic similarity evaluator - measures semantic similarity between texts.

        Returns:
            EvaluatorDetails configured for semantic similarity evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(slug="semantic-similarity", version=None, config=config)

    @staticmethod
    def agent_goal_accuracy(
    ) -> EvaluatorDetails:
        """
        Agent goal accuracy evaluator - validates agent goal achievement.

        Returns:
            EvaluatorDetails configured for agent goal accuracy evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(slug="agent-goal-accuracy", version=None, config=config)

    @staticmethod
    def topic_adherence(
    ) -> EvaluatorDetails:
        """
        Topic adherence evaluator - validates topic adherence.

        Returns:
            EvaluatorDetails configured for topic adherence evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(slug="topic-adherence", version=None, config=config)

    @staticmethod
    def perplexity(
    ) -> EvaluatorDetails:
        """
        Perplexity evaluator - measures text perplexity from logprobs.

        Returns:
            EvaluatorDetails configured for perplexity measurement
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(slug="perplexity", version=None, config=config)
