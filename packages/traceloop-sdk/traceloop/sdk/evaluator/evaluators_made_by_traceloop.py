from typing import Optional, Dict, Any, List
from .config import EvaluatorDetails


class EvaluatorMadeByTraceloop:
    """
    Factory class for creating made by traceloop evaluators with proper configuration.

    This class provides easy-to-use factory methods for all made by traceloop evaluators,
    with type hints and documentation for their configuration options.

    Example:
        >>> from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop
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

        Required task output fields:
            - text: The text to check for PII

        Args:
            probability_threshold: Minimum probability threshold for detecting PII (0.0-1.0)

        Returns:
            EvaluatorDetails configured for PII detection
        """
        config: Dict[str, Any] = {"probability_threshold": probability_threshold}
        return EvaluatorDetails(slug="pii-detector", version=None, config=config, required_input_fields=["text"])

    @staticmethod
    def toxicity_detector(
        threshold: float = 0.5,
    ) -> EvaluatorDetails:
        """
        Toxicity detector evaluator.

        Required task output fields:
            - text: The text to check for toxicity

        Args:
            threshold: Minimum toxicity threshold for flagging content (0.0-1.0)

        Returns:
            EvaluatorDetails configured for toxicity detection
        """
        config: Dict[str, Any] = {"threshold": threshold}

        return EvaluatorDetails(slug="toxicity-detector", version=None, config=config, required_input_fields=["text"])

    @staticmethod
    def prompt_injection(
        threshold: float = 0.5,
    ) -> EvaluatorDetails:
        """
        Prompt injection detector evaluator.

        Required task output fields:
            - prompt: The prompt to check for prompt injection attempts

        Args:
            threshold: Minimum threshold for detecting prompt injection attempts (0.0-1.0)

        Returns:
            EvaluatorDetails configured for prompt injection detection
        """
        config: Dict[str, Any] = {"threshold": threshold}
        return EvaluatorDetails(slug="prompt-injection", version=None, config=config, required_input_fields=["prompt"])

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

        Required task output fields:
            - text: The text to validate against the regex pattern

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

        return EvaluatorDetails(slug="regex-validator", version=None, config=config, required_input_fields=["text"])

    @staticmethod
    def json_validator(
        enable_schema_validation: bool = False,
        schema_string: Optional[str] = None,
    ) -> EvaluatorDetails:
        """
        JSON validator evaluator.

        Required task output fields:
            - text: The JSON text to validate

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

        return EvaluatorDetails(slug="json-validator", version=None, config=config, required_input_fields=["text"])

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

        Required task output fields:
            - text: The text to validate against the regex pattern
            - placeholder_value: The value of the placeholder to validate

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

        return EvaluatorDetails(
            slug="placeholder-regex",
            version=None,
            config=config,
            required_input_fields=["text", "placeholder_value"],
        )

    @staticmethod
    def char_count(
    ) -> EvaluatorDetails:
        """
        Character count evaluator - counts the number of characters in text.

        Required task output fields:
            - text: The text to count characters in

        Returns:
            EvaluatorDetails configured for character counting
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(slug="char-count", version=None, config=config, required_input_fields=["text"])

    @staticmethod
    def char_count_ratio(
    ) -> EvaluatorDetails:
        """
        Character count ratio evaluator - measures the ratio of characters between two texts.

        Required task output fields:
            - numerator_text: The numerator text for ratio calculation
            - denominator_text: The denominator text for ratio calculation

        Returns:
            EvaluatorDetails configured for character count ratio calculation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="char-count-ratio",
            version=None,
            config=config,
            required_input_fields=["numerator_text", "denominator_text"],
        )

    @staticmethod
    def word_count() -> EvaluatorDetails:
        """
        Word count evaluator - counts the number of words in text.

        Required task output fields:
            - text: The text to count words in

        Returns:
            EvaluatorDetails configured for word counting
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(slug="word-count", version=None, config=config, required_input_fields=["text"])

    @staticmethod
    def word_count_ratio(
    ) -> EvaluatorDetails:
        """
        Word count ratio evaluator - measures the ratio of words between two texts.

        Required task output fields:
           - numerator_text: The numerator text for ratio calculation
+          - denominator_text: The denominator text for ratio calculation

        Returns:
            EvaluatorDetails configured for word count ratio calculation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="word-count-ratio",
            version=None,
            config=config,
            required_input_fields=["numerator_text", "denominator_text"],
        )

    @staticmethod
    def answer_relevancy(
    ) -> EvaluatorDetails:
        """
        Answer relevancy evaluator - verifies responses address the query.

        Required task output fields:
            - question: The input question
            - answer: The answer to evaluate

        Returns:
            EvaluatorDetails configured for answer relevancy evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="answer-relevancy",
            version=None,
            config=config,
            required_input_fields=["question", "answer"],
        )

    @staticmethod
    def faithfulness(
    ) -> EvaluatorDetails:
        """
        Faithfulness evaluator - detects hallucinations and verifies facts.

        Required task output fields:
            - question: The input question
            - completion: The completion to evaluate for faithfulness
            - context: The context to verify against

        Returns:
            EvaluatorDetails configured for faithfulness evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="faithfulness",
            version=None,
            config=config,
            required_input_fields=["question", "completion", "context"],
        )

    @staticmethod
    def context_relevance(
    ) -> EvaluatorDetails:
        """
        Context relevance evaluator - validates context relevance.

        Required task output fields:
            - query: The user's query or question
            - context: The retrieved context to evaluate for relevance

        Returns:
            EvaluatorDetails configured for context relevance evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="context-relevance",
            version=None,
            config=config,
            required_input_fields=["query", "context"],
        )

    @staticmethod
    def profanity_detector() -> EvaluatorDetails:
        """
        Profanity detector evaluator - flags inappropriate language.

        Required task output fields:
            - text: The text to check for profanity

        Returns:
            EvaluatorDetails configured for profanity detection
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(slug="profanity-detector", version=None, config=config, required_input_fields=["text"])

    @staticmethod
    def sexism_detector(
        threshold: float = 0.5,
    ) -> EvaluatorDetails:
        """
        Sexism detector evaluator - detects sexist language and bias.

        Required task output fields:
            - text: The text to check for sexism

        Args:
            threshold: Minimum threshold for detecting sexism (0.0-1.0)

        Returns:
            EvaluatorDetails configured for sexism detection
        """
        config: Dict[str, Any] = {"threshold": threshold}

        return EvaluatorDetails(slug="sexism-detector", version=None, config=config, required_input_fields=["text"])

    @staticmethod
    def secrets_detector(
    ) -> EvaluatorDetails:
        """
        Secrets detector evaluator - monitors for credential and key leaks.

        Required task output fields:
            - text: The text to check for secrets

        Returns:
            EvaluatorDetails configured for secrets detection
        """
        config: Dict[str, Any] = {}
        return EvaluatorDetails(slug="secrets-detector", version=None, config=config, required_input_fields=["text"])

    @staticmethod
    def sql_validator(
    ) -> EvaluatorDetails:
        """
        SQL validator evaluator - validates SQL queries.

        Required task output fields:
            - text: The SQL query to validate

        Returns:
            EvaluatorDetails configured for SQL validation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(slug="sql-validator", version=None, config=config, required_input_fields=["text"])

    @staticmethod
    def semantic_similarity(
    ) -> EvaluatorDetails:
        """
        Semantic similarity evaluator - measures semantic similarity between texts.

        Required task output fields:
            - completion: The completion text to compare
            - reference: The reference text to compare against

        Returns:
            EvaluatorDetails configured for semantic similarity evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="semantic-similarity",
            version=None,
            config=config,
            required_input_fields=["completion", "reference"],
        )

    @staticmethod
    def agent_goal_accuracy(
    ) -> EvaluatorDetails:
        """
        Agent goal accuracy evaluator - validates agent goal achievement.

        Required task output fields:
            - question: The input question or goal
            - completion: The agent's completion
            - reference: The reference answer or goal

        Returns:
            EvaluatorDetails configured for agent goal accuracy evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="agent-goal-accuracy",
            version=None,
            config=config,
            required_input_fields=["question", "completion", "reference"],
        )

    @staticmethod
    def topic_adherence(
    ) -> EvaluatorDetails:
        """
        Topic adherence evaluator - validates topic adherence.

        Required task output fields:
            - question: The input question or goal
            - completion: The completion text to evaluate
            - reference_topics: The expected topic or topics

        Returns:
            EvaluatorDetails configured for topic adherence evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="topic-adherence",
            version=None,
            config=config,
            required_input_fields=["question", "completion", "reference_topics"],
        )

    @staticmethod
    def perplexity(
    ) -> EvaluatorDetails:
        """
        Perplexity evaluator - measures text perplexity from prompt.

        Required task output fields:
            - prompt: The prompt to measure perplexity for

        Returns:
            EvaluatorDetails configured for perplexity measurement
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="perplexity",
            version=None,
            config=config,
            required_input_fields=["prompt"],
        )

    @staticmethod
    def answer_completeness(
    ) -> EvaluatorDetails:
        """
        Answer completeness evaluator - measures how completely responses use relevant context.

        Required task output fields:
            - question: The input question
            - completion: The completion to evaluate
            - context: The context to evaluate against

        Returns:
            EvaluatorDetails configured for answer completeness evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="answer-completeness",
            version=None,
            config=config,
            required_input_fields=["question", "completion", "context"],
        )

    @staticmethod
    def answer_correctness(
    ) -> EvaluatorDetails:
        """
        Answer correctness evaluator - evaluates factual accuracy by comparing answers against ground truth.

        Required task output fields:
            - question: The input question
            - completion: The completion to evaluate
            - ground_truth: The ground truth answer

        Returns:
            EvaluatorDetails configured for answer correctness evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="answer-correctness",
            version=None,
            config=config,
            required_input_fields=["question", "completion", "ground_truth"],
        )

    @staticmethod
    def uncertainty_detector(
    ) -> EvaluatorDetails:
        """
        Uncertainty detector evaluator - generates responses and measures model uncertainty from logprobs.

        Required task output fields:
            - prompt: The prompt to evaluate uncertainty for

        Returns:
            EvaluatorDetails configured for uncertainty detection
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="uncertainty-detector",
            version=None,
            config=config,
            required_input_fields=["prompt"],
        )

    @staticmethod
    def agent_tool_error_detector(
    ) -> EvaluatorDetails:
        """
        Agent tool error detector evaluator - detects errors or failures during tool execution.

        Required task output fields:
            - tool_input: The input parameters passed to the tool
            - tool_output: The output or response from the tool execution

        Returns:
            EvaluatorDetails configured for agent tool error detection
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="agent-tool-error-detector",
            version=None,
            config=config,
            required_input_fields=["tool_input", "tool_output"],
        )

    @staticmethod
    def agent_flow_quality(
        threshold: float = 0.5,
        conditions: List[str] = [],
    ) -> EvaluatorDetails:
        """
        Agent flow quality evaluator - validates agent trajectories against user-defined natural language tests.

        Required task output fields:
            - trajectory_prompts: The prompts extracted from the span attributes (llm.prompts.*)
            - trajectory_completions: The completions extracted from the span attributes (llm.completions.*)
        Args:
            threshold: Minimum threshold for detecting tool errors (0.0-1.0)
            conditions: List of conditions in natural language to evaluate the agent flow quality against

        Returns:
            EvaluatorDetails configured for agent flow quality evaluation
        """
        config: Dict[str, Any] = {
            "threshold": threshold,
            "conditions": conditions,
        }

        return EvaluatorDetails(
            slug="agent-flow-quality",
            version=None,
            config=config,
            required_input_fields=["trajectory_prompts", "trajectory_completions"],
        )

    @staticmethod
    def agent_efficiency(
    ) -> EvaluatorDetails:
        """
        Agent efficiency evaluator - evaluates agent efficiency by checking for redundant calls and optimal paths.

        Required task output fields:
            - trajectory_prompts: The prompts extracted from the span attributes (llm.prompts.*)
            - trajectory_completions: The completions extracted from the span attributes (llm.completions.*)

        Returns:
            EvaluatorDetails configured for agent efficiency evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="agent-efficiency",
            version=None,
            config=config,
            required_input_fields=["trajectory_prompts", "trajectory_completions"],
        )

    @staticmethod
    def agent_goal_completeness(
    ) -> EvaluatorDetails:
        """
        Agent goal completeness evaluator - measures whether the agent successfully accomplished all user goals.

        Required task output fields:
            - trajectory_prompts: The prompts extracted from the span attributes (llm.prompts.*)
            - trajectory_completions: The completions extracted from the span attributes (llm.completions.*)

        Returns:
            EvaluatorDetails configured for agent goal completeness evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="agent-goal-completeness",
            version=None,
            config=config,
            required_input_fields=["trajectory_prompts", "trajectory_completions"],
        )

    @staticmethod
    def instruction_adherence(
    ) -> EvaluatorDetails:
        """
        Instruction adherence evaluator - measures how well the LLM response follows given instructions.

        Required task output fields:
            - instructions: The instructions to evaluate against
            - response: The response to evaluate

        Returns:
            EvaluatorDetails configured for instruction adherence evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="instruction-adherence",
            version=None,
            config=config,
            required_input_fields=["instructions", "response"],
        )

    @staticmethod
    def conversation_quality(
    ) -> EvaluatorDetails:
        """
        Conversation quality evaluator - evaluates conversation quality based on tone,
        clarity, flow, responsiveness, and transparency.

        Required task output fields:
            - prompts: The conversation prompts (flattened dict with llm.prompts.X.content/role)
            - completions: The conversation completions (flattened dict with llm.completions.X.content/role)

        Returns:
            EvaluatorDetails configured for conversation quality evaluation
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="conversation-quality",
            version=None,
            config=config,
            required_input_fields=["prompts", "completions"],
        )

    @staticmethod
    def intent_change(
    ) -> EvaluatorDetails:
        """
        Intent change evaluator - detects whether the user's primary intent or workflow
        changed significantly during a conversation.

        Required task output fields:
            - prompts: The conversation prompts (flattened dict with llm.prompts.X.content/role)
            - completions: The conversation completions (flattened dict with llm.completions.X.content/role)

        Returns:
            EvaluatorDetails configured for intent change detection
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="intent-change",
            version=None,
            config=config,
            required_input_fields=["prompts", "completions"],
        )

    @staticmethod
    def tone_detection(
    ) -> EvaluatorDetails:
        """
        Tone detection evaluator - classifies emotional tone of responses (joy, anger, sadness, etc.).

        Required task output fields:
            - text: The text to analyze for tone

        Returns:
            EvaluatorDetails configured for tone detection
        """
        config: Dict[str, Any] = {}

        return EvaluatorDetails(
            slug="tone-detection",
            version=None,
            config=config,
            required_input_fields=["text"],
        )
