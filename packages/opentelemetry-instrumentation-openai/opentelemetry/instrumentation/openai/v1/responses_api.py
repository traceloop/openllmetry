import json
import pydantic
import re
import time
from typing import Any, Optional, Union

from openai import AsyncStream, Stream
from openai._legacy_response import LegacyAPIResponse
from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
    openai_attributes as OpenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Span, Tracer
from typing_extensions import NotRequired
from wrapt import ObjectProxy

from opentelemetry.instrumentation.openai.shared import (
    _extract_model_name_from_provider_format,
    _set_request_attributes,
    _set_span_attribute,
    model_as_dict,
)
from opentelemetry.instrumentation.openai.utils import (
    _with_tracer_wrapper,
    dont_throw,
    should_send_prompts,
)
from opentelemetry.instrumentation.openai.v1.responses_runtime import (
    cache_legacy_parsed_response as _cache_legacy_parsed_response,
    handle_response_error,
    handle_response_success,
    prepare_response_request,
)
from opentelemetry.instrumentation.openai.v1.responses_stream_runtime import (
    ResponseStreamRuntime,
)


def _get_openai_sentinel_types() -> tuple:
    """Dynamically discover OpenAI sentinel types available in this SDK version.

    OpenAI SDK uses sentinel objects (NOT_GIVEN, Omit) for unset optional parameters.
    These types may not exist in older SDK versions, so we discover them at runtime.
    """
    sentinel_types = []
    try:
        from openai import NotGiven
        sentinel_types.append(NotGiven)
    except ImportError:
        pass
    try:
        from openai import Omit
        sentinel_types.append(Omit)
    except ImportError:
        pass
    return tuple(sentinel_types)


# Tuple of OpenAI sentinel types for isinstance() checks (empty if none available)
_OPENAI_SENTINEL_TYPES: tuple = _get_openai_sentinel_types()

# Conditional imports for backward compatibility
try:
    from openai.types.responses import (
        FunctionToolParam,
        Response,
        ResponseInputItemParam,
        ResponseInputParam,
        ResponseOutputItem,
        ResponseUsage,
        ToolParam,
    )
    from openai.types.responses.response_output_message_param import (
        ResponseOutputMessageParam,
    )
    RESPONSES_AVAILABLE = True
except ImportError:
    # Fallback types for older OpenAI SDK versions
    from typing import Dict, List

    # Create basic fallback types
    FunctionToolParam = Dict[str, Any]
    Response = Any
    ResponseInputItemParam = Dict[str, Any]
    ResponseInputParam = Union[str, List[Dict[str, Any]]]
    ResponseOutputItem = Dict[str, Any]
    ResponseUsage = Dict[str, Any]
    ToolParam = Dict[str, Any]
    ResponseOutputMessageParam = Dict[str, Any]
    RESPONSES_AVAILABLE = False

SPAN_NAME = "openai.response"


def _sanitize_sentinel_values(kwargs: dict) -> dict:
    """Remove OpenAI sentinel values (NOT_GIVEN, Omit) from kwargs.

    OpenAI SDK uses sentinel objects for unset optional parameters.
    These don't have dict methods like .get(), causing errors when
    code chains calls like kwargs.get("reasoning", {}).get("summary").

    This removes sentinel values so the default (e.g., {}) is used instead
    when calling .get() on the sanitized dict.

    If no sentinel types are available (older SDK), returns kwargs unchanged.
    """
    if not _OPENAI_SENTINEL_TYPES:
        return kwargs
    return {k: v for k, v in kwargs.items()
            if not isinstance(v, _OPENAI_SENTINEL_TYPES)}


def prepare_input_param(input_param: ResponseInputItemParam) -> ResponseInputItemParam:
    """
    Looks like OpenAI API infers the type "message" if the shape is correct,
    but type is not specified.
    It is marked as required on the message types. We add this to our
    traced data to make it work.
    """
    try:
        d = model_as_dict(input_param)
        if "type" not in d:
            d["type"] = "message"
        if RESPONSES_AVAILABLE:
            return ResponseInputItemParam(**d)
        else:
            return d
    except Exception:
        return input_param


def process_input(inp: ResponseInputParam) -> ResponseInputParam:
    if not isinstance(inp, list):
        return inp
    return [prepare_input_param(item) for item in inp]


def is_validator_iterator(content):
    """
    Some OpenAI objects contain fields typed as Iterable, which pydantic
    internally converts to a ValidatorIterator, and they cannot be trivially
    serialized without consuming the iterator to, for example, a list.

    See: https://github.com/pydantic/pydantic/issues/9541#issuecomment-2189045051
    """
    return re.search(r"pydantic.*ValidatorIterator'>$", str(type(content)))


# OpenAI API accepts output messages without an ID in its inputs, but
# the ID is marked as required in the output type.
if RESPONSES_AVAILABLE:
    class ResponseOutputMessageParamWithoutId(ResponseOutputMessageParam):
        id: NotRequired[str]
else:
    # Fallback for older SDK versions
    ResponseOutputMessageParamWithoutId = dict


class TracedData(pydantic.BaseModel):
    start_time: float  # time.time_ns()
    response_id: str
    # actually Union[str, list[Union[ResponseInputItemParam, ResponseOutputMessageParamWithoutId]]],
    # but this only works properly in Python 3.10+ / newer pydantic
    input: Any
    # system message
    instructions: Optional[str] = pydantic.Field(default=None)
    # TODO: remove Any with newer Python / pydantic
    tools: Optional[list[Union[Any, ToolParam]]] = pydantic.Field(default=None)
    output_blocks: Optional[dict[str, ResponseOutputItem]] = pydantic.Field(
        default=None
    )
    usage: Optional[ResponseUsage] = pydantic.Field(default=None)
    output_text: Optional[str] = pydantic.Field(default=None)
    request_model: Optional[str] = pydantic.Field(default=None)
    response_model: Optional[str] = pydantic.Field(default=None)

    # Reasoning attributes
    request_reasoning_summary: Optional[str] = pydantic.Field(default=None)
    request_reasoning_effort: Optional[str] = pydantic.Field(default=None)
    response_reasoning_effort: Optional[str] = pydantic.Field(default=None)

    # OpenAI service tier
    request_service_tier: Optional[str] = pydantic.Field(default=None)
    response_service_tier: Optional[str] = pydantic.Field(default=None)

    # Trace context - to maintain trace continuity across async operations
    trace_context: Any = pydantic.Field(default=None)

    class Config:
        arbitrary_types_allowed = True


responses: dict[str, TracedData] = {}


def parse_response(response: Union[LegacyAPIResponse, Response]) -> Response:
    if isinstance(response, LegacyAPIResponse):
        return response.parse()
    return response


def get_tools_from_kwargs(kwargs: dict) -> list[ToolParam]:
    tools_input = kwargs.get("tools", [])
    # Handle case where tools key exists but value is None
    # (e.g., when wrappers like openai-guardrails pass tools=None)
    if tools_input is None:
        tools_input = []
    tools = []

    for tool in tools_input:
        if tool.get("type") == "function":
            if RESPONSES_AVAILABLE:
                tools.append(FunctionToolParam(**tool))
            else:
                tools.append(tool)

    return tools


def process_content_block(
    block: dict[str, Any],
) -> dict[str, Any]:
    # TODO: keep the original type once backend supports it
    if block.get("type") in ["text", "input_text", "output_text"]:
        return {"type": "text", "text": block.get("text")}
    elif block.get("type") in ["image", "input_image", "output_image"]:
        return {
            "type": "image",
            "image_url": block.get("image_url"),
            "detail": block.get("detail"),
            "file_id": block.get("file_id"),
        }
    elif block.get("type") in ["file", "input_file", "output_file"]:
        return {
            "type": "file",
            "file_id": block.get("file_id"),
            "filename": block.get("filename"),
            "file_data": block.get("file_data"),
        }
    return block


@dont_throw
def prepare_kwargs_for_shared_attributes(kwargs):
    """
    Prepare kwargs for the shared _set_request_attributes function.
    Maps responses API specific parameters to the common format.
    """
    prepared_kwargs = kwargs.copy()

    # Map max_output_tokens to max_tokens for the shared function
    if "max_output_tokens" in kwargs:
        prepared_kwargs["max_tokens"] = kwargs["max_output_tokens"]

    return prepared_kwargs


def set_data_attributes(traced_response: TracedData, span: Span):
    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_ID, traced_response.response_id)

    response_model = _extract_model_name_from_provider_format(traced_response.response_model)
    _set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, response_model)

    _set_span_attribute(span, OpenAIAttributes.OPENAI_RESPONSE_SERVICE_TIER, traced_response.response_service_tier)
    if usage := traced_response.usage:
        _set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, usage.input_tokens)
        _set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, usage.output_tokens)
        _set_span_attribute(
            span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total_tokens
        )
        if usage.input_tokens_details:
            _set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
                usage.input_tokens_details.cached_tokens,
            )

        reasoning_tokens = None
        tokens_details = (
            usage.get("output_tokens_details") if isinstance(usage, dict)
            else getattr(usage, "output_tokens_details", None)
        )

        if tokens_details:
            reasoning_tokens = (
                tokens_details.get("reasoning_tokens", None) if isinstance(tokens_details, dict)
                else getattr(tokens_details, "reasoning_tokens", None)
            )

        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_REASONING_TOKENS,
            reasoning_tokens or 0,
        )

    request_reasoning_summary_attr = getattr(
        SpanAttributes, "LLM_REQUEST_REASONING_SUMMARY", None
    )
    request_reasoning_effort_attr = getattr(
        SpanAttributes, "LLM_REQUEST_REASONING_EFFORT", None
    )
    response_reasoning_effort_attr = getattr(
        SpanAttributes, "LLM_RESPONSE_REASONING_EFFORT", None
    )
    if request_reasoning_summary_attr:
        _set_span_attribute(
            span,
            request_reasoning_summary_attr,
            traced_response.request_reasoning_summary or (),
        )
    if request_reasoning_effort_attr:
        _set_span_attribute(
            span,
            request_reasoning_effort_attr,
            traced_response.request_reasoning_effort or (),
        )
    if response_reasoning_effort_attr:
        _set_span_attribute(
            span,
            response_reasoning_effort_attr,
            traced_response.response_reasoning_effort or (),
        )

    if should_send_prompts():
        prompt_index = 0
        if traced_response.tools:
            for i, tool_param in enumerate(traced_response.tools):
                tool_dict = model_as_dict(tool_param)
                description = tool_dict.get("description")
                parameters = tool_dict.get("parameters")
                name = tool_dict.get("name")
                if parameters is None:
                    continue
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}.description",
                    description,
                )
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}.parameters",
                    json.dumps(parameters),
                )
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}.name",
                    name,
                )
        if traced_response.instructions:
            _set_span_attribute(
                span,
                f"{GenAIAttributes.GEN_AI_PROMPT}.{prompt_index}.content",
                traced_response.instructions,
            )
            _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_PROMPT}.{prompt_index}.role", "system")
            prompt_index += 1

        if isinstance(traced_response.input, str):
            _set_span_attribute(
                span, f"{GenAIAttributes.GEN_AI_PROMPT}.{prompt_index}.content", traced_response.input
            )
            _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_PROMPT}.{prompt_index}.role", "user")
            prompt_index += 1
        else:
            for block in traced_response.input:
                block_dict = model_as_dict(block)
                if block_dict.get("type", "message") == "message":
                    content = block_dict.get("content")
                    if is_validator_iterator(content):
                        # we're after the actual call here, so we can consume the iterator
                        content = [process_content_block(block) for block in content]
                    try:
                        stringified_content = (
                            content if isinstance(content, str) else json.dumps(content)
                        )
                    except Exception:
                        stringified_content = (
                            str(content) if content is not None else ""
                        )
                    _set_span_attribute(
                        span,
                        f"{GenAIAttributes.GEN_AI_PROMPT}.{prompt_index}.content",
                        stringified_content,
                    )
                    _set_span_attribute(
                        span,
                        f"{GenAIAttributes.GEN_AI_PROMPT}.{prompt_index}.role",
                        block_dict.get("role"),
                    )
                    prompt_index += 1
                elif block_dict.get("type") == "computer_call_output":
                    _set_span_attribute(
                        span, f"{GenAIAttributes.GEN_AI_PROMPT}.{prompt_index}.role", "computer-call"
                    )
                    output_image_url = block_dict.get("output", {}).get("image_url")
                    if output_image_url:
                        _set_span_attribute(
                            span,
                            f"{GenAIAttributes.GEN_AI_PROMPT}.{prompt_index}.content",
                            json.dumps(
                                [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": output_image_url},
                                    }
                                ]
                            ),
                        )
                    prompt_index += 1
                elif block_dict.get("type") == "computer_call":
                    _set_span_attribute(
                        span, f"{GenAIAttributes.GEN_AI_PROMPT}.{prompt_index}.role", "assistant"
                    )
                    call_content = {}
                    if block_dict.get("id"):
                        call_content["id"] = block_dict.get("id")
                    if block_dict.get("call_id"):
                        call_content["call_id"] = block_dict.get("call_id")
                    if block_dict.get("action"):
                        call_content["action"] = block_dict.get("action")
                    _set_span_attribute(
                        span,
                        f"{GenAIAttributes.GEN_AI_PROMPT}.{prompt_index}.content",
                        json.dumps(call_content),
                    )
                    prompt_index += 1
                # TODO: handle other block types

        _set_span_attribute(span, f"{GenAIAttributes.GEN_AI_COMPLETION}.0.role", "assistant")
        if traced_response.output_text:
            _set_span_attribute(
                span, f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content", traced_response.output_text
            )
        tool_call_index = 0
        for block in traced_response.output_blocks.values():
            block_dict = model_as_dict(block)
            if block_dict.get("type") == "message":
                # either a refusal or handled in output_text above
                continue
            if block_dict.get("type") == "function_call":
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.id",
                    block_dict.get("id"),
                )
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.name",
                    block_dict.get("name"),
                )
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.arguments",
                    block_dict.get("arguments"),
                )
                tool_call_index += 1
            elif block_dict.get("type") == "file_search_call":
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.id",
                    block_dict.get("id"),
                )
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.name",
                    "file_search_call",
                )
                tool_call_index += 1
            elif block_dict.get("type") == "web_search_call":
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.id",
                    block_dict.get("id"),
                )
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.name",
                    "web_search_call",
                )
                tool_call_index += 1
            elif block_dict.get("type") == "computer_call":
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.id",
                    block_dict.get("call_id"),
                )
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.name",
                    "computer_call",
                )
                _set_span_attribute(
                    span,
                    f"{GenAIAttributes.GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.arguments",
                    json.dumps(block_dict.get("action")),
                )
                tool_call_index += 1
            elif block_dict.get("type") == "reasoning":
                reasoning_summary = block_dict.get("summary")
                if reasoning_summary is not None and reasoning_summary != []:
                    if isinstance(reasoning_summary, (dict, list)):
                        reasoning_value = json.dumps(reasoning_summary)
                    else:
                        reasoning_value = reasoning_summary
                    _set_span_attribute(
                        span, f"{GenAIAttributes.GEN_AI_COMPLETION}.0.reasoning", reasoning_value
                    )
            # TODO: handle other block types, in particular other calls


@dont_throw
@_with_tracer_wrapper
def responses_get_or_create_wrapper(tracer: Tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)
    start_time = time.time_ns()

    # Remove OpenAI sentinel values (NOT_GIVEN, Omit) to allow chained .get() calls
    ctx = context_api.get_current()
    non_sentinel_kwargs = _sanitize_sentinel_values(kwargs)
    span, non_sentinel_kwargs = prepare_response_request(
        tracer=tracer,
        span_name=SPAN_NAME,
        start_time=start_time,
        context=ctx,
        request_kwargs=non_sentinel_kwargs,
        instance=instance,
        prepare_request_attributes=prepare_kwargs_for_shared_attributes,
        set_request_attributes=_set_request_attributes,
    )

    try:
        response = wrapped(*args, **non_sentinel_kwargs)
        if isinstance(response, Stream):
            return ResponseStream(
                span=span,
                response=response,
                start_time=start_time,
                request_kwargs=non_sentinel_kwargs,
                tracer=tracer,
                instance=instance,
            )
    except Exception as e:
        handle_response_error(
            tracer=tracer,
            span_name=SPAN_NAME,
            error=e,
            current_span=span,
            context=ctx,
            request_kwargs=non_sentinel_kwargs,
            instance=instance,
            start_time=start_time,
            traced_data_cls=TracedData,
            response_store=responses,
            process_input=process_input,
            get_tools_from_kwargs=get_tools_from_kwargs,
            prepare_request_attributes=prepare_kwargs_for_shared_attributes,
            set_request_attributes=_set_request_attributes,
            set_data_attributes=set_data_attributes,
        )
        raise
    parsed_response = parse_response(response)
    if not handle_response_success(
        tracer=tracer,
        span_name=SPAN_NAME,
        current_span=span,
        context=ctx,
        request_kwargs=non_sentinel_kwargs,
        instance=instance,
        start_time=start_time,
        response=response,
        parsed_response=parsed_response,
        traced_data_cls=TracedData,
        response_store=responses,
        process_input=process_input,
        get_tools_from_kwargs=get_tools_from_kwargs,
        prepare_request_attributes=prepare_kwargs_for_shared_attributes,
        set_request_attributes=_set_request_attributes,
        set_data_attributes=set_data_attributes,
    ):
        return response
    return response


@dont_throw
@_with_tracer_wrapper
async def async_responses_get_or_create_wrapper(
    tracer: Tracer, wrapped, instance, args, kwargs
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)
    start_time = time.time_ns()

    # Remove OpenAI sentinel values (NOT_GIVEN, Omit) to allow chained .get() calls
    ctx = context_api.get_current()
    non_sentinel_kwargs = _sanitize_sentinel_values(kwargs)
    span, non_sentinel_kwargs = prepare_response_request(
        tracer=tracer,
        span_name=SPAN_NAME,
        start_time=start_time,
        context=ctx,
        request_kwargs=non_sentinel_kwargs,
        instance=instance,
        prepare_request_attributes=prepare_kwargs_for_shared_attributes,
        set_request_attributes=_set_request_attributes,
    )

    try:
        response = await wrapped(*args, **non_sentinel_kwargs)
        if isinstance(response, (Stream, AsyncStream)):
            return ResponseStream(
                span=span,
                response=response,
                start_time=start_time,
                request_kwargs=non_sentinel_kwargs,
                tracer=tracer,
                instance=instance,
            )
    except Exception as e:
        handle_response_error(
            tracer=tracer,
            span_name=SPAN_NAME,
            error=e,
            current_span=span,
            context=ctx,
            request_kwargs=non_sentinel_kwargs,
            instance=instance,
            start_time=start_time,
            traced_data_cls=TracedData,
            response_store=responses,
            process_input=process_input,
            get_tools_from_kwargs=get_tools_from_kwargs,
            prepare_request_attributes=prepare_kwargs_for_shared_attributes,
            set_request_attributes=_set_request_attributes,
            set_data_attributes=set_data_attributes,
        )
        raise
    parsed_response = parse_response(response)
    if not handle_response_success(
        tracer=tracer,
        span_name=SPAN_NAME,
        current_span=span,
        context=ctx,
        request_kwargs=non_sentinel_kwargs,
        instance=instance,
        start_time=start_time,
        response=response,
        parsed_response=parsed_response,
        traced_data_cls=TracedData,
        response_store=responses,
        process_input=process_input,
        get_tools_from_kwargs=get_tools_from_kwargs,
        prepare_request_attributes=prepare_kwargs_for_shared_attributes,
        set_request_attributes=_set_request_attributes,
        set_data_attributes=set_data_attributes,
    ):
        return response
    return response


@dont_throw
@_with_tracer_wrapper
def responses_cancel_wrapper(tracer: Tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    non_sentinel_kwargs = _sanitize_sentinel_values(kwargs)

    response = wrapped(*args, **kwargs)
    if isinstance(response, Stream):
        return response
    parsed_response = parse_response(response)
    existing_data = responses.pop(parsed_response.id, None)
    if existing_data is not None:
        # Restore the original trace context to maintain trace continuity
        ctx = existing_data.trace_context if existing_data.trace_context else context_api.get_current()
        span = tracer.start_span(
            SPAN_NAME,
            kind=SpanKind.CLIENT,
            start_time=existing_data.start_time,
            record_exception=True,
            context=ctx,
        )
        _set_request_attributes(span, prepare_kwargs_for_shared_attributes(non_sentinel_kwargs), instance)
        span.record_exception(Exception("Response cancelled"))
        set_data_attributes(existing_data, span)
        span.end()
    return response


@dont_throw
@_with_tracer_wrapper
async def async_responses_cancel_wrapper(
    tracer: Tracer, wrapped, instance, args, kwargs
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    non_sentinel_kwargs = _sanitize_sentinel_values(kwargs)

    response = await wrapped(*args, **kwargs)
    if isinstance(response, (Stream, AsyncStream)):
        return response
    parsed_response = parse_response(response)
    existing_data = responses.pop(parsed_response.id, None)
    if existing_data is not None:
        # Restore the original trace context to maintain trace continuity
        ctx = existing_data.trace_context if existing_data.trace_context else context_api.get_current()
        span = tracer.start_span(
            SPAN_NAME,
            kind=SpanKind.CLIENT,
            start_time=existing_data.start_time,
            record_exception=True,
            context=ctx,
        )
        _set_request_attributes(span, prepare_kwargs_for_shared_attributes(non_sentinel_kwargs), instance)
        span.record_exception(Exception("Response cancelled"))
        set_data_attributes(existing_data, span)
        span.end()
    return response


class ResponseStream(ObjectProxy):
    """Proxy class for streaming responses to capture telemetry data"""

    _span = None
    _start_time = None
    _request_kwargs = None
    _tracer = None
    _traced_data = None

    def __init__(
        self,
        span,
        response,
        start_time=None,
        request_kwargs=None,
        tracer=None,
        traced_data=None,
        instance=None,
    ):
        super().__init__(response)
        self._runtime = ResponseStreamRuntime(
            self,
            span=span,
            start_time=start_time,
            request_kwargs=_sanitize_sentinel_values(request_kwargs or {}),
            tracer=tracer,
            traced_data=traced_data,
            instance=instance,
            traced_data_cls=TracedData,
            response_store=responses,
            span_name=SPAN_NAME,
            process_input=process_input,
            get_tools_from_kwargs=get_tools_from_kwargs,
            prepare_request_attributes=prepare_kwargs_for_shared_attributes,
            set_request_attributes=_set_request_attributes,
            parse_response=parse_response,
            set_data_attributes=set_data_attributes,
            cache_legacy_parsed_response=_cache_legacy_parsed_response,
        )

    def __del__(self):
        """Cleanup when object is garbage collected"""
        if hasattr(self, "_cleanup_completed") and not self._cleanup_completed:
            self._runtime.ensure_cleanup()

    def __enter__(self):
        """Context manager entry"""
        if hasattr(self.__wrapped__, "__enter__"):
            self.__wrapped__.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        suppress = False
        try:
            if exc_type is not None:
                self._runtime.handle_exception(exc_val)
            else:
                self._runtime.process_complete_response()
        finally:
            if hasattr(self.__wrapped__, "__exit__"):
                suppress = bool(self.__wrapped__.__exit__(exc_type, exc_val, exc_tb))
        return suppress

    async def __aenter__(self):
        """Async context manager entry"""
        if hasattr(self.__wrapped__, "__aenter__"):
            await self.__wrapped__.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        suppress = False
        try:
            if exc_type is not None:
                self._runtime.handle_exception(exc_val)
            else:
                self._runtime.process_complete_response()
        finally:
            if hasattr(self.__wrapped__, "__aexit__"):
                suppress = bool(await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb))
        return suppress

    def close(self):
        try:
            self._runtime.ensure_cleanup()
        finally:
            if hasattr(self.__wrapped__, "close"):
                return self.__wrapped__.close()

    async def aclose(self):
        try:
            self._runtime.ensure_cleanup()
        finally:
            if hasattr(self.__wrapped__, "aclose"):
                return await self.__wrapped__.aclose()

    def __iter__(self):
        """Synchronous iterator"""
        return self

    def __next__(self):
        """Synchronous iteration"""
        return self._runtime.next()

    def __aiter__(self):
        """Async iterator"""
        return self

    async def __anext__(self):
        """Async iteration"""
        return await self._runtime.anext()

    def _process_chunk(self, chunk):
        """Process a streaming chunk"""
        self._runtime.process_chunk(chunk)

    @dont_throw
    def _process_complete_response(self):
        """Process the complete response and emit span"""
        self._runtime.process_complete_response()

    @dont_throw
    def _handle_exception(self, exception):
        """Handle exceptions during streaming"""
        self._runtime.handle_exception(exception)

    @dont_throw
    def _ensure_cleanup(self):
        """Ensure cleanup happens even if stream is not fully consumed"""
        self._runtime.ensure_cleanup()
