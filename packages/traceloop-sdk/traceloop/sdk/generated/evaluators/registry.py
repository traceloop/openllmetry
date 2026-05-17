"""
Registry mapping evaluator slugs to their request/response Pydantic models.

This enables type-safe validation of inputs and parsing of outputs.

DO NOT EDIT MANUALLY - Regenerate with:
    ./scripts/generate-models.sh /path/to/swagger.json
"""

from typing import Dict, Type, Optional
from pydantic import BaseModel

from .request import (
    AgentEfficiencyRequest,
    AgentFlowQualityRequest,
    AgentGoalAccuracyRequest,
    AgentGoalCompletenessRequest,
    AgentToolErrorDetectorRequest,
    AgentToolTrajectoryRequest,
    AnswerCompletenessRequest,
    AnswerCorrectnessRequest,
    AnswerRelevancyRequest,
    CharCountRatioRequest,
    CharCountRequest,
    ContextRelevanceRequest,
    ConversationQualityRequest,
    FaithfulnessRequest,
    HtmlComparisonRequest,
    InstructionAdherenceRequest,
    IntentChangeRequest,
    JSONValidatorRequest,
    PIIDetectorRequest,
    PerplexityRequest,
    PlaceholderRegexRequest,
    ProfanityDetectorRequest,
    PromptInjectionRequest,
    PromptPerplexityRequest,
    RegexValidatorRequest,
    SQLValidatorRequest,
    SecretsDetectorRequest,
    SemanticSimilarityRequest,
    SexismDetectorRequest,
    ToneDetectionRequest,
    TopicAdherenceRequest,
    ToxicityDetectorRequest,
    UncertaintyDetectorRequest,
    WordCountRatioRequest,
    WordCountRequest,
)

from .response import (
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
    PIIDetectorResponse,
    PerplexityResponse,
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


# Mapping from evaluator slug to request model
REQUEST_MODELS: Dict[str, Type[BaseModel]] = {
    "agent-efficiency": AgentEfficiencyRequest,
    "agent-flow-quality": AgentFlowQualityRequest,
    "agent-goal-accuracy": AgentGoalAccuracyRequest,
    "agent-goal-completeness": AgentGoalCompletenessRequest,
    "agent-tool-error-detector": AgentToolErrorDetectorRequest,
    "agent-tool-trajectory": AgentToolTrajectoryRequest,
    "answer-completeness": AnswerCompletenessRequest,
    "answer-correctness": AnswerCorrectnessRequest,
    "answer-relevancy": AnswerRelevancyRequest,
    "char-count": CharCountRequest,
    "char-count-ratio": CharCountRatioRequest,
    "context-relevance": ContextRelevanceRequest,
    "conversation-quality": ConversationQualityRequest,
    "faithfulness": FaithfulnessRequest,
    "html-comparison": HtmlComparisonRequest,
    "instruction-adherence": InstructionAdherenceRequest,
    "intent-change": IntentChangeRequest,
    "json-validator": JSONValidatorRequest,
    "perplexity": PerplexityRequest,
    "pii-detector": PIIDetectorRequest,
    "placeholder-regex": PlaceholderRegexRequest,
    "profanity-detector": ProfanityDetectorRequest,
    "prompt-injection": PromptInjectionRequest,
    "prompt-perplexity": PromptPerplexityRequest,
    "regex-validator": RegexValidatorRequest,
    "secrets-detector": SecretsDetectorRequest,
    "semantic-similarity": SemanticSimilarityRequest,
    "sexism-detector": SexismDetectorRequest,
    "sql-validator": SQLValidatorRequest,
    "tone-detection": ToneDetectionRequest,
    "topic-adherence": TopicAdherenceRequest,
    "toxicity-detector": ToxicityDetectorRequest,
    "uncertainty-detector": UncertaintyDetectorRequest,
    "word-count": WordCountRequest,
    "word-count-ratio": WordCountRatioRequest,
}

# Mapping from evaluator slug to response model
RESPONSE_MODELS: Dict[str, Type[BaseModel]] = {
    "agent-efficiency": AgentEfficiencyResponse,
    "agent-flow-quality": AgentFlowQualityResponse,
    "agent-goal-accuracy": AgentGoalAccuracyResponse,
    "agent-goal-completeness": AgentGoalCompletenessResponse,
    "agent-tool-error-detector": AgentToolErrorDetectorResponse,
    "agent-tool-trajectory": AgentToolTrajectoryResponse,
    "answer-completeness": AnswerCompletenessResponse,
    "answer-correctness": AnswerCorrectnessResponse,
    "answer-relevancy": AnswerRelevancyResponse,
    "char-count": CharCountResponse,
    "char-count-ratio": CharCountRatioResponse,
    "context-relevance": ContextRelevanceResponse,
    "conversation-quality": ConversationQualityResponse,
    "faithfulness": FaithfulnessResponse,
    "html-comparison": HtmlComparisonResponse,
    "instruction-adherence": InstructionAdherenceResponse,
    "intent-change": IntentChangeResponse,
    "json-validator": JSONValidatorResponse,
    "perplexity": PerplexityResponse,
    "pii-detector": PIIDetectorResponse,
    "placeholder-regex": PlaceholderRegexResponse,
    "profanity-detector": ProfanityDetectorResponse,
    "prompt-injection": PromptInjectionResponse,
    "prompt-perplexity": PromptPerplexityResponse,
    "regex-validator": RegexValidatorResponse,
    "secrets-detector": SecretsDetectorResponse,
    "semantic-similarity": SemanticSimilarityResponse,
    "sexism-detector": SexismDetectorResponse,
    "sql-validator": SQLValidatorResponse,
    "tone-detection": ToneDetectionResponse,
    "topic-adherence": TopicAdherenceResponse,
    "toxicity-detector": ToxicityDetectorResponse,
    "uncertainty-detector": UncertaintyDetectorResponse,
    "word-count": WordCountResponse,
    "word-count-ratio": WordCountRatioResponse,
}


def get_request_model(slug: str) -> Optional[Type[BaseModel]]:
    """Get the request model for an evaluator by slug."""
    return REQUEST_MODELS.get(slug)


def get_response_model(slug: str) -> Optional[Type[BaseModel]]:
    """Get the response model for an evaluator by slug."""
    return RESPONSE_MODELS.get(slug)
