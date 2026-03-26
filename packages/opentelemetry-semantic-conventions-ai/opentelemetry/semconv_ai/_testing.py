"""
Shared compliance test classes for opentelemetry-semantic-conventions-ai.

Import these classes in any instrumentation package's test suite to verify
that the installed semconv constants have the expected values:

    from opentelemetry.semconv_ai._testing import *  # noqa: F401, F403

pytest will discover and run all Test* classes that end up in the module
namespace, so a single import line is enough.
"""

import pytest
from opentelemetry.semconv_ai import GenAISystem, Meters, SpanAttributes


# ---------------------------------------------------------------------------
# SpanAttributes — renamed constants (LLM_* → GEN_AI_*)
# ---------------------------------------------------------------------------


class TestSpanAttributesGENAIRenamed:
    """Verify all renamed LLM_* → GEN_AI_* constants have the correct string values."""

    def test_gen_ai_usage_total_tokens(self):
        assert SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS == "gen_ai.usage.total_tokens"

    def test_gen_ai_usage_token_type(self):
        assert SpanAttributes.GEN_AI_USAGE_TOKEN_TYPE == "gen_ai.usage.token_type"

    def test_gen_ai_user(self):
        assert SpanAttributes.GEN_AI_USER == "gen_ai.user"

    def test_gen_ai_headers(self):
        assert SpanAttributes.GEN_AI_HEADERS == "gen_ai.headers"

    def test_gen_ai_is_streaming(self):
        assert SpanAttributes.GEN_AI_IS_STREAMING == "gen_ai.is_streaming"

    def test_gen_ai_request_repetition_penalty(self):
        assert SpanAttributes.GEN_AI_REQUEST_REPETITION_PENALTY == "gen_ai.request.repetition_penalty"

    def test_gen_ai_response_finish_reason(self):
        assert SpanAttributes.GEN_AI_RESPONSE_FINISH_REASON == "gen_ai.response.finish_reason"

    def test_gen_ai_response_stop_reason(self):
        assert SpanAttributes.GEN_AI_RESPONSE_STOP_REASON == "gen_ai.response.stop_reason"

    def test_gen_ai_content_completion_chunk(self):
        assert SpanAttributes.GEN_AI_CONTENT_COMPLETION_CHUNK == "gen_ai.content.completion.chunk"

    def test_gen_ai_request_reasoning_effort(self):
        assert SpanAttributes.GEN_AI_REQUEST_REASONING_EFFORT == "gen_ai.request.reasoning_effort"

    def test_gen_ai_usage_reasoning_tokens(self):
        assert SpanAttributes.GEN_AI_USAGE_REASONING_TOKENS == "gen_ai.usage.reasoning_tokens"

    def test_gen_ai_request_n(self):
        assert SpanAttributes.GEN_AI_REQUEST_N == "gen_ai.request.n"

    def test_gen_ai_request_max_completion_tokens(self):
        assert SpanAttributes.GEN_AI_REQUEST_MAX_COMPLETION_TOKENS == "gen_ai.request.max_completion_tokens"

    def test_gen_ai_request_structured_output_schema(self):
        assert SpanAttributes.GEN_AI_REQUEST_STRUCTURED_OUTPUT_SCHEMA == "gen_ai.request.structured_output_schema"

    def test_gen_ai_request_reasoning_summary(self):
        assert SpanAttributes.GEN_AI_REQUEST_REASONING_SUMMARY == "gen_ai.request.reasoning_summary"

    def test_gen_ai_response_reasoning_effort(self):
        assert SpanAttributes.GEN_AI_RESPONSE_REASONING_EFFORT == "gen_ai.response.reasoning_effort"

    def test_gen_ai_openai_api_base(self):
        assert SpanAttributes.GEN_AI_OPENAI_API_BASE == "gen_ai.openai.api_base"

    def test_gen_ai_openai_api_version(self):
        assert SpanAttributes.GEN_AI_OPENAI_API_VERSION == "gen_ai.openai.api_version"

    def test_gen_ai_openai_api_type(self):
        assert SpanAttributes.GEN_AI_OPENAI_API_TYPE == "gen_ai.openai.api_type"


# ---------------------------------------------------------------------------
# SpanAttributes — old LLM_* names must be gone
# ---------------------------------------------------------------------------


class TestSpanAttributesLegacyLLMNamesPresent:
    """Assert that legacy LLM_* constants are still available for non-migrated packages."""

    @pytest.mark.parametrize(
        "legacy_name,expected_value",
        [
            ("LLM_SYSTEM", "gen_ai.system"),
            ("LLM_REQUEST_MODEL", "gen_ai.request.model"),
            ("LLM_REQUEST_MAX_TOKENS", "gen_ai.request.max_tokens"),
            ("LLM_REQUEST_TEMPERATURE", "gen_ai.request.temperature"),
            ("LLM_REQUEST_TOP_P", "gen_ai.request.top_p"),
            ("LLM_PROMPTS", "gen_ai.prompt"),
            ("LLM_COMPLETIONS", "gen_ai.completion"),
            ("LLM_RESPONSE_MODEL", "gen_ai.response.model"),
            ("LLM_USAGE_COMPLETION_TOKENS", "gen_ai.usage.completion_tokens"),
            ("LLM_USAGE_PROMPT_TOKENS", "gen_ai.usage.prompt_tokens"),
            ("LLM_USAGE_CACHE_CREATION_INPUT_TOKENS", "gen_ai.usage.cache_creation_input_tokens"),
            ("LLM_USAGE_CACHE_READ_INPUT_TOKENS", "gen_ai.usage.cache_read_input_tokens"),
            ("LLM_TOKEN_TYPE", "gen_ai.token.type"),
            ("LLM_REQUEST_TYPE", "llm.request.type"),
            ("LLM_FREQUENCY_PENALTY", "llm.frequency_penalty"),
            ("LLM_PRESENCE_PENALTY", "llm.presence_penalty"),
            ("LLM_CHAT_STOP_SEQUENCES", "llm.chat.stop_sequences"),
            ("LLM_REQUEST_FUNCTIONS", "llm.request.functions"),
            ("LLM_TOP_K", "llm.top_k"),
            ("LLM_OPENAI_RESPONSE_SYSTEM_FINGERPRINT", "gen_ai.openai.system_fingerprint"),
            ("LLM_IS_STREAMING", "llm.is_streaming"),
            ("LLM_USAGE_TOTAL_TOKENS", "llm.usage.total_tokens"),
            ("LLM_USER", "llm.user"),
            ("LLM_HEADERS", "llm.headers"),
            ("LLM_RESPONSE_FINISH_REASON", "llm.response.finish_reason"),
            ("LLM_RESPONSE_STOP_REASON", "llm.response.stop_reason"),
            ("LLM_CONTENT_COMPLETION_CHUNK", "llm.content.completion.chunk"),
            ("LLM_REQUEST_REASONING_EFFORT", "llm.request.reasoning_effort"),
            ("LLM_USAGE_REASONING_TOKENS", "llm.usage.reasoning_tokens"),
            ("LLM_USAGE_TOKEN_TYPE", "llm.usage.token_type"),
            ("LLM_REQUEST_REPETITION_PENALTY", "llm.request.repetition_penalty"),
            ("LLM_REQUEST_STRUCTURED_OUTPUT_SCHEMA", "gen_ai.request.structured_output_schema"),
            ("LLM_REQUEST_REASONING_SUMMARY", "gen_ai.request.reasoning_summary"),
            ("LLM_RESPONSE_REASONING_EFFORT", "gen_ai.response.reasoning_effort"),
            ("LLM_OPENAI_API_BASE", "gen_ai.openai.api_base"),
            ("LLM_OPENAI_API_VERSION", "gen_ai.openai.api_version"),
            ("LLM_OPENAI_API_TYPE", "gen_ai.openai.api_type"),
            ("LLM_DECODING_METHOD", "llm.watsonx.decoding_method"),
            ("LLM_RANDOM_SEED", "llm.watsonx.random_seed"),
            ("LLM_MAX_NEW_TOKENS", "llm.watsonx.max_new_tokens"),
            ("LLM_MIN_NEW_TOKENS", "llm.watsonx.min_new_tokens"),
            ("LLM_REPETITION_PENALTY", "llm.watsonx.repetition_penalty"),
        ],
    )
    def test_legacy_name_present_with_old_value(self, legacy_name, expected_value):
        assert hasattr(SpanAttributes, legacy_name), (
            f"SpanAttributes.{legacy_name} must exist for backward compatibility."
        )
        assert getattr(SpanAttributes, legacy_name) == expected_value


# ---------------------------------------------------------------------------
# SpanAttributes — cache attributes
# ---------------------------------------------------------------------------


class TestSpanAttributesCacheDotSeparator:
    """Cache token attributes use dot-separated sub-namespaces (spec update)."""

    def test_gen_ai_usage_cache_read_input_tokens(self):
        assert SpanAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS == "gen_ai.usage.cache_read.input_tokens"

    def test_gen_ai_usage_cache_creation_input_tokens(self):
        assert SpanAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS == "gen_ai.usage.cache_creation.input_tokens"


# ---------------------------------------------------------------------------
# SpanAttributes — project-policy attributes use gen_ai namespace
# ---------------------------------------------------------------------------


class TestSpanAttributesProjectPolicy:
    """Project-policy attributes (not in upstream OTel spec) use gen_ai namespace."""

    def test_is_streaming(self):
        assert SpanAttributes.GEN_AI_IS_STREAMING == "gen_ai.is_streaming"

    def test_user(self):
        assert SpanAttributes.GEN_AI_USER == "gen_ai.user"

    def test_headers(self):
        assert SpanAttributes.GEN_AI_HEADERS == "gen_ai.headers"


class TestSpanAttributesOldValuesAbsent:
    """Regression: old/incorrect string values must not appear anywhere in SpanAttributes."""

    @pytest.mark.parametrize(
        "old_value",
        [
            "llm.usage.total_tokens",
            "llm.frequency_penalty",
            "llm.presence_penalty",
            "llm.is_streaming",
            "llm.user",
            "llm.headers",
            "llm.top_k",
            "llm.chat.stop_sequences",
            "llm.request.functions",
            "llm.request.repetition_penalty",
            "llm.request.type",
            "llm.usage.token_type",
            "llm.response.finish_reason",
            "llm.response.stop_reason",
            "llm.content.completion.chunk",
            "llm.request.reasoning_effort",
            "llm.usage.reasoning_tokens",
            "llm.chat_completions.streaming_time_to_generate",
            "gen_ai.usage.cache_read_input_tokens",    # underscore variant (pre-migration)
            "gen_ai.usage.cache_creation_input_tokens",  # underscore variant (pre-migration)
        ],
    )
    def test_old_value_not_in_new_span_attributes(self, old_value):
        all_values = {
            name: value
            for name, value in vars(SpanAttributes).items()
            if not name.startswith("_") and isinstance(value, str)
            and not name.startswith("LLM_")  # exclude legacy aliases
            and not name.endswith("_DEPRECATED")  # exclude deprecated aliases
        }
        assert old_value not in all_values.values(), (
            f"Old attribute value {old_value!r} is still present in a GEN_AI_* SpanAttribute. "
            f"It should have been renamed."
        )


class TestSpanAttributesUnchanged:
    """Constants that should NOT have changed — sanity check."""

    def test_traceloop_span_kind_unchanged(self):
        assert SpanAttributes.TRACELOOP_SPAN_KIND == "traceloop.span.kind"


# ---------------------------------------------------------------------------
# SpanAttributes — Watsonx vendor-specific attributes (renamed to GEN_AI_WATSONX_*)
# ---------------------------------------------------------------------------


class TestSpanAttributesWatsonxKept:
    """
    llm.watsonx.* span attributes are intentionally kept. These use llm.watsonx as a
    vendor-qualified prefix (analogous to db.chroma.*), not a generic llm.* namespace.
    The Python names have been renamed to GEN_AI_WATSONX_* prefix.
    """

    def test_watsonx_decoding_method_kept(self):
        assert SpanAttributes.GEN_AI_WATSONX_DECODING_METHOD == "llm.watsonx.decoding_method"

    def test_watsonx_random_seed_kept(self):
        assert SpanAttributes.GEN_AI_WATSONX_RANDOM_SEED == "llm.watsonx.random_seed"

    def test_watsonx_max_new_tokens_kept(self):
        assert SpanAttributes.GEN_AI_WATSONX_MAX_NEW_TOKENS == "llm.watsonx.max_new_tokens"

    def test_watsonx_min_new_tokens_kept(self):
        assert SpanAttributes.GEN_AI_WATSONX_MIN_NEW_TOKENS == "llm.watsonx.min_new_tokens"

    def test_watsonx_repetition_penalty_kept(self):
        assert SpanAttributes.GEN_AI_WATSONX_REPETITION_PENALTY == "llm.watsonx.repetition_penalty"


# ---------------------------------------------------------------------------
# GenAISystem enum — values must match OTel GenAiSystemValues where possible
# ---------------------------------------------------------------------------


class TestGenAISystemOtelAligned:
    """Enum members that have a counterpart in OTel GenAiSystemValues."""

    def test_openai(self):
        assert GenAISystem.OPENAI.value == "openai"

    def test_anthropic_lowercase(self):
        # Was "Anthropic" — must now match OTel GenAiSystemValues.ANTHROPIC
        from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GenAiSystemValues

        assert GenAISystem.ANTHROPIC.value == GenAiSystemValues.ANTHROPIC.value
        assert GenAISystem.ANTHROPIC.value == "anthropic"

    def test_cohere_lowercase(self):
        from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GenAiSystemValues

        assert GenAISystem.COHERE.value == GenAiSystemValues.COHERE.value
        assert GenAISystem.COHERE.value == "cohere"

    def test_mistralai_spec_format(self):
        from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GenAiSystemValues

        assert GenAISystem.MISTRALAI.value == GenAiSystemValues.MISTRAL_AI.value
        assert GenAISystem.MISTRALAI.value == "mistral_ai"

    def test_groq_lowercase(self):
        from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GenAiSystemValues

        assert GenAISystem.GROQ.value == GenAiSystemValues.GROQ.value
        assert GenAISystem.GROQ.value == "groq"

    def test_watsonx_spec_format(self):
        from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GenAiSystemValues

        assert GenAISystem.WATSONX.value == GenAiSystemValues.IBM_WATSONX_AI.value
        assert GenAISystem.WATSONX.value == "ibm.watsonx.ai"

    def test_aws_spec_format(self):
        from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GenAiSystemValues

        assert GenAISystem.AWS.value == GenAiSystemValues.AWS_BEDROCK.value
        assert GenAISystem.AWS.value == "aws.bedrock"

    def test_azure_spec_format(self):
        from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GenAiSystemValues

        assert GenAISystem.AZURE.value == GenAiSystemValues.AZ_AI_OPENAI.value
        assert GenAISystem.AZURE.value == "az.ai.openai"

    def test_google_spec_format(self):
        from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GenAiSystemValues

        assert GenAISystem.GOOGLE.value == GenAiSystemValues.GCP_GEN_AI.value
        assert GenAISystem.GOOGLE.value == "gcp.gen_ai"


class TestGenAISystemProjectValues:
    """Enum members without an OTel counterpart — project-defined lowercase values."""

    def test_ollama(self):
        assert GenAISystem.OLLAMA.value == "ollama"

    def test_aleph_alpha(self):
        assert GenAISystem.ALEPH_ALPHA.value == "aleph_alpha"

    def test_replicate(self):
        assert GenAISystem.REPLICATE.value == "replicate"

    def test_together_ai(self):
        assert GenAISystem.TOGETHER_AI.value == "together_ai"

    def test_huggingface(self):
        assert GenAISystem.HUGGINGFACE.value == "hugging_face"

    def test_fireworks(self):
        assert GenAISystem.FIREWORKS.value == "fireworks"

    def test_openrouter(self):
        assert GenAISystem.OPENROUTER.value == "openrouter"

    def test_langchain(self):
        assert GenAISystem.LANGCHAIN.value == "langchain"

    def test_crewai(self):
        assert GenAISystem.CREWAI.value == "crewai"


class TestGenAISystemNoCaps:
    """All GenAISystem values must be lowercase (no PascalCase or camelCase)."""

    def test_all_values_lowercase(self):
        non_lowercase = [
            member.name
            for member in GenAISystem
            if member.value != member.value.lower() and "." not in member.value
        ]
        assert non_lowercase == [], (
            f"GenAISystem members have non-lowercase values: {non_lowercase}. "
            "Values should use lowercase with dots or underscores."
        )


# ---------------------------------------------------------------------------
# Meters — metric names must use gen_ai.* namespace
# ---------------------------------------------------------------------------


class TestMetersGenAiNamespace:
    """Generic metric names must use gen_ai.* namespace."""

    def test_streaming_time_to_generate(self):
        assert Meters.LLM_STREAMING_TIME_TO_GENERATE == "llm.chat_completions.streaming_time_to_generate"

    def test_core_metrics_unchanged(self):
        """Core gen_ai.client.* metrics already had the correct namespace."""
        assert Meters.LLM_GENERATION_CHOICES == "gen_ai.client.generation.choices"
        assert Meters.LLM_TOKEN_USAGE == "gen_ai.client.token.usage"
        assert Meters.LLM_OPERATION_DURATION == "gen_ai.client.operation.duration"


class TestMetersVendorNamespacesKept:
    """
    Vendor-qualified metric names (llm.openai.*, llm.anthropic.*, llm.watsonx.*)
    are intentionally kept. The llm.<vendor> prefix is a vendor identifier, not the
    generic llm.* attribute namespace being migrated. These will be renamed in the
    respective package PRs if/when those vendors adopt the gen_ai namespace.
    """

    def test_openai_completions_exceptions_kept(self):
        assert Meters.LLM_COMPLETIONS_EXCEPTIONS == "llm.openai.chat_completions.exceptions"

    def test_openai_embeddings_exceptions_kept(self):
        assert Meters.LLM_EMBEDDINGS_EXCEPTIONS == "llm.openai.embeddings.exceptions"

    def test_openai_embeddings_vector_size_kept(self):
        assert Meters.LLM_EMBEDDINGS_VECTOR_SIZE == "llm.openai.embeddings.vector_size"

    def test_openai_image_generations_exceptions_kept(self):
        assert Meters.LLM_IMAGE_GENERATIONS_EXCEPTIONS == "llm.openai.image_generations.exceptions"

    def test_anthropic_completion_exceptions_kept(self):
        assert Meters.LLM_ANTHROPIC_COMPLETION_EXCEPTIONS == "llm.anthropic.completion.exceptions"

    def test_watsonx_metrics_kept(self):
        assert Meters.LLM_WATSONX_COMPLETIONS_DURATION == "llm.watsonx.completions.duration"
        assert Meters.LLM_WATSONX_COMPLETIONS_EXCEPTIONS == "llm.watsonx.completions.exceptions"
        assert Meters.LLM_WATSONX_COMPLETIONS_RESPONSES == "llm.watsonx.completions.responses"
        assert Meters.LLM_WATSONX_COMPLETIONS_TOKENS == "llm.watsonx.completions.tokens"
