"""OpenTelemetry Google Generative AI API instrumentation"""

from collections import defaultdict
import logging
import os
from typing import AsyncGenerator, Callable, Collection, Generator, Optional

from google.genai import types

from opentelemetry.instrumentation.google_genai.config import Config
from opentelemetry.instrumentation.google_genai.utils import (
    dont_throw,
    role_from_content_union,
    set_span_attribute,
    text_from_content_union,
    to_dict,
    with_tracer_wrapper,
)
from opentelemetry.trace import Tracer
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind, Span
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap

from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    LLMRequestTypeValues,
)
from opentelemetry.instrumentation.google_genai.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("google-genai >= 1.0.0",)

WRAPPED_METHODS = [
    {
        "package": "google.genai.models",
        "object": "Models",
        "method": "generate_content",
        "span_name": "gemini.generate_content",
        "is_streaming": False,
        "is_async": False,
    },
    {
        "package": "google.genai.models",
        "object": "AsyncModels",
        "method": "generate_content",
        "span_name": "gemini.generate_content",
        "is_streaming": False,
        "is_async": True,
    },
    {
        "package": "google.genai.models",
        "object": "Models",
        "method": "generate_content_stream",
        "span_name": "gemini.generate_content_stream",
        "is_streaming": True,
        "is_async": False,
    },
    {
        "package": "google.genai.models",
        "object": "AsyncModels",
        "method": "generate_content_stream",
        "span_name": "gemini.generate_content_stream",
        "is_streaming": True,
        "is_async": True,
    }
]


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


@dont_throw
def _set_request_attributes(span, args, kwargs):
    config_dict = to_dict(kwargs.get("config", {}))
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_MODEL,
        kwargs.get("model")
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_TEMPERATURE,
        config_dict.get("temperature")
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_TOP_P,
        config_dict.get("top_p")
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_TOP_K,
        config_dict.get("top_k")
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_CHOICE_COUNT,
        config_dict.get("candidate_count")
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_MAX_TOKENS,
        config_dict.get("max_output_tokens")
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_STOP_SEQUENCES,
        config_dict.get("stop_sequences")
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_FREQUENCY_PENALTY,
        config_dict.get("frequency_penalty")
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_PRESENCE_PENALTY,
        config_dict.get("presence_penalty")
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_SEED,
        config_dict.get("seed")
    )

    tools: list[types.FunctionDeclaration] = []
    if kwargs.get("tools"):
        for tool in kwargs.get("tools"):
            if isinstance(tool, types.Tool):
                tools += tool.function_declarations or []
            elif isinstance(tool, Callable):
                tools.append(types.FunctionDeclaration.from_callable(tool))
    for tool_num, tool in enumerate(tools):
        set_span_attribute(
            span,
            f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{tool_num}.name",
            to_dict(tool).get("name")
        )
        set_span_attribute(
            span,
            f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{tool_num}.description",
            to_dict(tool).get("description")
        )
        set_span_attribute(
            span,
            f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{tool_num}.parameters",
            to_dict(tool).get("parameters")
        )

    if should_send_prompts():
        i = 0
        system_instruction: Optional[types.ContentUnion] = config_dict.get("system_instruction")
        if system_instruction:
            set_span_attribute(
                span,
                f"{gen_ai_attributes.GEN_AI_PROMPT}.{i}.content",
                text_from_content_union(system_instruction)
            )
            set_span_attribute(span, f"{gen_ai_attributes.GEN_AI_PROMPT}.{i}.role", "system")
            i += 1
        contents = kwargs.get("contents", [])
        if not isinstance(contents, list):
            contents = [contents]
        for content in contents:
            set_span_attribute(
                span,
                f"{gen_ai_attributes.GEN_AI_PROMPT}.{i}.content",
                text_from_content_union(content)
            )
            set_span_attribute(
                span,
                f"{gen_ai_attributes.GEN_AI_PROMPT}.{i}.role",
                role_from_content_union(content) or "user"
            )
            i += 1


@dont_throw
def _set_response_attributes(span, response: types.GenerateContentResponse):
    candidates = response.candidates or []
    set_span_attribute(span, gen_ai_attributes.GEN_AI_RESPONSE_ID, to_dict(response).get("response_id"))
    set_span_attribute(span, gen_ai_attributes.GEN_AI_RESPONSE_MODEL, to_dict(response).get("model_version"))

    if response.usage_metadata:
        usage_dict = to_dict(response.usage_metadata)
        set_span_attribute(
            span,
            gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS,
            usage_dict.get("prompt_token_count")
        )
        set_span_attribute(
            span,
            gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS,
            usage_dict.get("candidates_token_count")
        )
        set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            usage_dict.get("total_token_count")
        )
        set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            usage_dict.get("cached_content_token_count")
        )

    if should_send_prompts():
        if len(candidates) > 1:
            for i, candidate in enumerate(candidates):
                set_span_attribute(
                    span,
                    f"{gen_ai_attributes.GEN_AI_COMPLETION}.{i}.content",
                    text_from_content_union(candidate.content)
                )
                set_span_attribute(span, f"{gen_ai_attributes.GEN_AI_COMPLETION}.{i}.role", "assistant")
        else:
            set_span_attribute(span, f"{gen_ai_attributes.GEN_AI_COMPLETION}.0.content", response.text)
            set_span_attribute(span, f"{gen_ai_attributes.GEN_AI_COMPLETION}.0.role", "assistant")


@dont_throw
def _build_from_streaming_response(
    span: Span,
    response: Generator[types.GenerateContentResponse, None, None]
) -> Generator[types.GenerateContentResponse, None, None]:
    final_parts = []
    role = "model"
    aggregated_usage_metadata = defaultdict(int)
    model_version = None
    for chunk in response:
        if chunk.model_version:
            model_version = chunk.model_version

        if chunk.candidates:
            # Currently gemini throws an error if you pass more than one candidate
            # with streaming
            if chunk.candidates and len(chunk.candidates) > 0:
                final_parts += (chunk.candidates[0].content.parts or [])
                role = chunk.candidates[0].content.role or role
        if chunk.usage_metadata:
            usage_dict = to_dict(chunk.usage_metadata)
            # prompt token count is sent in every chunk
            # (and is less by 1 in the last chunk, so we set it once);
            # total token count in every chunk is greater by prompt token count than it should be,
            # thus this awkward logic here
            if aggregated_usage_metadata.get("prompt_token_count") is None:
                aggregated_usage_metadata["prompt_token_count"] = usage_dict.get("prompt_token_count") or 0
                aggregated_usage_metadata["total_token_count"] = usage_dict.get("total_token_count") or 0
            aggregated_usage_metadata["candidates_token_count"] += usage_dict.get("candidates_token_count") or 0
            aggregated_usage_metadata["total_token_count"] += usage_dict.get("candidates_token_count") or 0
        yield chunk

    compound_response = types.GenerateContentResponse(
        candidates=[{
            "content": {
                "parts": final_parts,
                "role": role,
            },
        }],
        usage_metadata=types.GenerateContentResponseUsageMetadataDict(**aggregated_usage_metadata),
        model_version=model_version,
    )
    if span.is_recording():
        _set_response_attributes(span, compound_response)
    span.end()


@dont_throw
async def _abuild_from_streaming_response(
    span: Span,
    response: AsyncGenerator[types.GenerateContentResponse, None]
) -> AsyncGenerator[types.GenerateContentResponse, None]:
    final_parts = []
    role = "model"
    aggregated_usage_metadata = defaultdict(int)
    model_version = None
    async for chunk in response:
        if chunk.candidates:
            # Currently gemini throws an error if you pass more than one candidate
            # with streaming
            if chunk.candidates and len(chunk.candidates) > 0:
                final_parts += (chunk.candidates[0].content.parts or [])
                role = chunk.candidates[0].content.role or role
                if chunk.model_version:
                    model_version = chunk.model_version
        if chunk.usage_metadata:
            usage_dict = to_dict(chunk.usage_metadata)
            # prompt token count is sent in every chunk
            # (and is less by 1 in the last chunk, so we set it once);
            # total token count in every chunk is greater by prompt token count than it should be,
            # thus this awkward logic here
            if aggregated_usage_metadata.get("prompt_token_count") is None:
                aggregated_usage_metadata["prompt_token_count"] = usage_dict.get("prompt_token_count")
                aggregated_usage_metadata["total_token_count"] = usage_dict.get("total_token_count")
            aggregated_usage_metadata["candidates_token_count"] += usage_dict.get("candidates_token_count") or 0
            aggregated_usage_metadata["total_token_count"] += usage_dict.get("candidates_token_count") or 0
        yield chunk

    compound_response = types.GenerateContentResponse(
        candidates=[{
            "content": {
                "parts": final_parts,
                "role": role,
            },
        }],
        usage_metadata=types.GenerateContentResponseUsageMetadataDict(**aggregated_usage_metadata),
        model_version=model_version,
    )
    if span.is_recording():
        _set_response_attributes(span, compound_response)
    span.end()


@with_tracer_wrapper
def _wrap(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    span = tracer.start_span(
        to_wrap.get("span_name"),
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "gemini",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )

    if span.is_recording():
        _set_request_attributes(span, args, kwargs)

    if to_wrap.get("is_streaming"):
        return _build_from_streaming_response(span, wrapped(*args, **kwargs))
    else:
        response = wrapped(*args, **kwargs)

    if span.is_recording():
        _set_response_attributes(span, response)

    span.end()
    return response


@with_tracer_wrapper
async def _awrap(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    span = tracer.start_span(
        to_wrap.get("span_name"),
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "gemini",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
    )

    if span.is_recording():
        _set_request_attributes(span, args, kwargs)

    if to_wrap.get("is_streaming"):
        return _abuild_from_streaming_response(span, await wrapped(*args, **kwargs))
    else:
        response = await wrapped(*args, **kwargs)

    if span.is_recording():
        _set_response_attributes(span, response)

    span.end()
    return response


class GoogleGenAiInstrumentor(BaseInstrumentor):
    """An instrumentor for Google GenAI's client library."""

    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        for wrapped_method in WRAPPED_METHODS:
            wrap_function_wrapper(
                wrapped_method.get("package"),
                f"{wrapped_method.get('object')}.{wrapped_method.get('method')}",
                (
                    _awrap(tracer, wrapped_method)
                    if wrapped_method.get("is_async")
                    else _wrap(tracer, wrapped_method)
                ),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            unwrap(
                f"{wrapped_method.get('package')}.{wrapped_method.get('object')}",
                wrapped_method.get("method")
            )
