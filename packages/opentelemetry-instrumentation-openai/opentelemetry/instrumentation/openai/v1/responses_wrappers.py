import json
import pydantic
import re
import threading
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
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import SpanKind, Span, StatusCode, Tracer
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

    _set_span_attribute(
        span,
        f"{SpanAttributes.LLM_REQUEST_REASONING_SUMMARY}",
        traced_response.request_reasoning_summary or (),
    )
    _set_span_attribute(
        span,
        f"{SpanAttributes.LLM_REQUEST_REASONING_EFFORT}",
        traced_response.request_reasoning_effort or (),
    )
    _set_span_attribute(
        span,
        f"{SpanAttributes.LLM_RESPONSE_REASONING_EFFORT}",
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
    non_sentinel_kwargs = _sanitize_sentinel_values(kwargs)

    try:
        response = wrapped(*args, **kwargs)
        if isinstance(response, Stream):
            # Capture current trace context to maintain trace continuity
            ctx = context_api.get_current()
            span = tracer.start_span(
                SPAN_NAME,
                kind=SpanKind.CLIENT,
                start_time=start_time,
                context=ctx,
            )
            _set_request_attributes(span, prepare_kwargs_for_shared_attributes(non_sentinel_kwargs), instance)

            return ResponseStream(
                span=span,
                response=response,
                start_time=start_time,
                request_kwargs=non_sentinel_kwargs,
                tracer=tracer,
            )
    except Exception as e:
        response_id = non_sentinel_kwargs.get("response_id")
        existing_data = {}
        if response_id and response_id in responses:
            existing_data = responses[response_id].model_dump()
        try:
            traced_data = TracedData(
                start_time=existing_data.get("start_time", start_time),
                response_id=response_id or "",
                input=process_input(
                    non_sentinel_kwargs.get("input", existing_data.get("input", []))
                ),
                instructions=non_sentinel_kwargs.get(
                    "instructions", existing_data.get("instructions")
                ),
                tools=get_tools_from_kwargs(non_sentinel_kwargs) or existing_data.get("tools", []),
                output_blocks=existing_data.get("output_blocks", {}),
                usage=existing_data.get("usage"),
                output_text=non_sentinel_kwargs.get(
                    "output_text", existing_data.get("output_text", "")
                ),
                request_model=non_sentinel_kwargs.get(
                    "model", existing_data.get("request_model", "")
                ),
                response_model=existing_data.get("response_model", ""),
                # Reasoning attributes
                request_reasoning_summary=(
                    non_sentinel_kwargs.get("reasoning", {}).get(
                        "summary", existing_data.get("request_reasoning_summary")
                    )
                ),
                request_reasoning_effort=(
                    non_sentinel_kwargs.get("reasoning", {}).get(
                        "effort", existing_data.get("request_reasoning_effort")
                    )
                ),
                response_reasoning_effort=non_sentinel_kwargs.get("reasoning", {}).get("effort"),
                request_service_tier=non_sentinel_kwargs.get("service_tier"),
                response_service_tier=existing_data.get("response_service_tier"),
                # Capture trace context to maintain continuity
                trace_context=existing_data.get("trace_context", context_api.get_current()),
            )
        except Exception:
            traced_data = None

        # Restore the original trace context to maintain trace continuity
        ctx = (traced_data.trace_context if traced_data and traced_data.trace_context
               else context_api.get_current())
        span = tracer.start_span(
            SPAN_NAME,
            kind=SpanKind.CLIENT,
            start_time=(
                start_time if traced_data is None else int(traced_data.start_time)
            ),
            context=ctx,
        )
        _set_request_attributes(span, prepare_kwargs_for_shared_attributes(non_sentinel_kwargs), instance)
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e)
        span.set_status(StatusCode.ERROR, str(e))
        if traced_data:
            set_data_attributes(traced_data, span)
        span.end()
        raise
    parsed_response = parse_response(response)

    existing_data = responses.get(parsed_response.id)
    if existing_data is None:
        existing_data = {}
    else:
        existing_data = existing_data.model_dump()

    request_tools = get_tools_from_kwargs(non_sentinel_kwargs)

    merged_tools = existing_data.get("tools", []) + request_tools

    try:
        parsed_response_output_text = None
        if hasattr(parsed_response, "output_text"):
            parsed_response_output_text = parsed_response.output_text
        else:
            try:
                parsed_response_output_text = parsed_response.output[0].content[0].text
            except Exception:
                pass
        traced_data = TracedData(
            start_time=existing_data.get("start_time", start_time),
            response_id=parsed_response.id,
            input=process_input(existing_data.get("input", non_sentinel_kwargs.get("input"))),
            instructions=existing_data.get("instructions", non_sentinel_kwargs.get("instructions")),
            tools=merged_tools if merged_tools else None,
            output_blocks={block.id: block for block in parsed_response.output}
            | existing_data.get("output_blocks", {}),
            usage=existing_data.get("usage", parsed_response.usage),
            output_text=existing_data.get("output_text", parsed_response_output_text),
            request_model=existing_data.get("request_model", non_sentinel_kwargs.get("model")),
            response_model=existing_data.get("response_model", parsed_response.model),
            # Reasoning attributes
            request_reasoning_summary=(
                non_sentinel_kwargs.get("reasoning", {}).get(
                    "summary", existing_data.get("request_reasoning_summary")
                )
            ),
            request_reasoning_effort=(
                non_sentinel_kwargs.get("reasoning", {}).get(
                    "effort", existing_data.get("request_reasoning_effort")
                )
            ),
            response_reasoning_effort=non_sentinel_kwargs.get("reasoning", {}).get("effort"),
            request_service_tier=existing_data.get("request_service_tier", non_sentinel_kwargs.get("service_tier")),
            response_service_tier=existing_data.get("response_service_tier", parsed_response.service_tier),
            # Capture trace context to maintain continuity across async operations
            trace_context=existing_data.get("trace_context", context_api.get_current()),
        )
        responses[parsed_response.id] = traced_data
    except Exception:
        return response

    if parsed_response.status == "completed":
        # Restore the original trace context to maintain trace continuity
        ctx = traced_data.trace_context if traced_data.trace_context else context_api.get_current()
        span = tracer.start_span(
            SPAN_NAME,
            kind=SpanKind.CLIENT,
            start_time=int(traced_data.start_time),
            context=ctx,
        )
        _set_request_attributes(span, prepare_kwargs_for_shared_attributes(non_sentinel_kwargs), instance)
        set_data_attributes(traced_data, span)
        span.end()

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
    non_sentinel_kwargs = _sanitize_sentinel_values(kwargs)

    try:
        response = await wrapped(*args, **kwargs)
        if isinstance(response, (Stream, AsyncStream)):
            # Capture current trace context to maintain trace continuity
            ctx = context_api.get_current()
            span = tracer.start_span(
                SPAN_NAME,
                kind=SpanKind.CLIENT,
                start_time=start_time,
                context=ctx,
            )
            _set_request_attributes(span, prepare_kwargs_for_shared_attributes(non_sentinel_kwargs), instance)

            return ResponseStream(
                span=span,
                response=response,
                start_time=start_time,
                request_kwargs=non_sentinel_kwargs,
                tracer=tracer,
            )
    except Exception as e:
        response_id = non_sentinel_kwargs.get("response_id")
        existing_data = {}
        if response_id and response_id in responses:
            existing_data = responses[response_id].model_dump()
        try:
            traced_data = TracedData(
                start_time=existing_data.get("start_time", start_time),
                response_id=response_id or "",
                input=process_input(
                    non_sentinel_kwargs.get("input", existing_data.get("input", []))
                ),
                instructions=non_sentinel_kwargs.get(
                    "instructions", existing_data.get("instructions", "")
                ),
                tools=get_tools_from_kwargs(non_sentinel_kwargs) or existing_data.get("tools", []),
                output_blocks=existing_data.get("output_blocks", {}),
                usage=existing_data.get("usage"),
                output_text=non_sentinel_kwargs.get("output_text", existing_data.get("output_text")),
                request_model=non_sentinel_kwargs.get("model", existing_data.get("request_model")),
                response_model=existing_data.get("response_model"),
                # Reasoning attributes
                request_reasoning_summary=(
                    non_sentinel_kwargs.get("reasoning", {}).get(
                        "summary", existing_data.get("request_reasoning_summary")
                    )
                ),
                request_reasoning_effort=(
                    non_sentinel_kwargs.get("reasoning", {}).get(
                        "effort", existing_data.get("request_reasoning_effort")
                    )
                ),
                response_reasoning_effort=non_sentinel_kwargs.get("reasoning", {}).get("effort"),
                request_service_tier=non_sentinel_kwargs.get("service_tier"),
                response_service_tier=existing_data.get("response_service_tier"),
                # Capture trace context to maintain continuity
                trace_context=existing_data.get("trace_context", context_api.get_current()),
            )
        except Exception:
            traced_data = None

        # Restore the original trace context to maintain trace continuity
        ctx = (traced_data.trace_context if traced_data and traced_data.trace_context
               else context_api.get_current())
        span = tracer.start_span(
            SPAN_NAME,
            kind=SpanKind.CLIENT,
            start_time=(
                start_time if traced_data is None else int(traced_data.start_time)
            ),
            context=ctx,
        )
        _set_request_attributes(span, prepare_kwargs_for_shared_attributes(non_sentinel_kwargs), instance)
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e)
        span.set_status(StatusCode.ERROR, str(e))
        if traced_data:
            set_data_attributes(traced_data, span)
        span.end()
        raise
    parsed_response = parse_response(response)

    existing_data = responses.get(parsed_response.id)
    if existing_data is None:
        existing_data = {}
    else:
        existing_data = existing_data.model_dump()

    request_tools = get_tools_from_kwargs(non_sentinel_kwargs)

    merged_tools = existing_data.get("tools", []) + request_tools

    try:
        parsed_response_output_text = None
        if hasattr(parsed_response, "output_text"):
            parsed_response_output_text = parsed_response.output_text
        else:
            try:
                parsed_response_output_text = parsed_response.output[0].content[0].text
            except Exception:
                pass

        traced_data = TracedData(
            start_time=existing_data.get("start_time", start_time),
            response_id=parsed_response.id,
            input=process_input(existing_data.get("input", non_sentinel_kwargs.get("input"))),
            instructions=existing_data.get("instructions", non_sentinel_kwargs.get("instructions")),
            tools=merged_tools if merged_tools else None,
            output_blocks={block.id: block for block in parsed_response.output}
            | existing_data.get("output_blocks", {}),
            usage=existing_data.get("usage", parsed_response.usage),
            output_text=existing_data.get("output_text", parsed_response_output_text),
            request_model=existing_data.get("request_model", non_sentinel_kwargs.get("model")),
            response_model=existing_data.get("response_model", parsed_response.model),
            # Reasoning attributes
            request_reasoning_summary=(
                non_sentinel_kwargs.get("reasoning", {}).get(
                    "summary", existing_data.get("request_reasoning_summary")
                )
            ),
            request_reasoning_effort=(
                non_sentinel_kwargs.get("reasoning", {}).get(
                    "effort", existing_data.get("request_reasoning_effort")
                )
            ),
            response_reasoning_effort=non_sentinel_kwargs.get("reasoning", {}).get("effort"),
            request_service_tier=existing_data.get("request_service_tier", non_sentinel_kwargs.get("service_tier")),
            response_service_tier=existing_data.get("response_service_tier", parsed_response.service_tier),
            # Capture trace context to maintain continuity across async operations
            trace_context=existing_data.get("trace_context", context_api.get_current()),
        )
        responses[parsed_response.id] = traced_data
    except Exception:
        return response

    if parsed_response.status == "completed":
        # Restore the original trace context to maintain trace continuity
        ctx = traced_data.trace_context if traced_data.trace_context else context_api.get_current()
        span = tracer.start_span(
            SPAN_NAME,
            kind=SpanKind.CLIENT,
            start_time=int(traced_data.start_time),
            context=ctx,
        )
        _set_request_attributes(span, prepare_kwargs_for_shared_attributes(non_sentinel_kwargs), instance)
        set_data_attributes(traced_data, span)
        span.end()

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
    ):
        super().__init__(response)
        self._span = span
        self._start_time = start_time
        # Filter sentinel values (defensive, in case called directly without prior filtering)
        self._request_kwargs = _sanitize_sentinel_values(request_kwargs or {})
        self._tracer = tracer
        self._traced_data = traced_data or TracedData(
            start_time=start_time,
            response_id="",
            input=process_input(self._request_kwargs.get("input", [])),
            instructions=self._request_kwargs.get("instructions"),
            tools=get_tools_from_kwargs(self._request_kwargs),
            output_blocks={},
            usage=None,
            output_text="",
            request_model=self._request_kwargs.get("model", ""),
            response_model="",
            request_reasoning_summary=self._request_kwargs.get("reasoning", {}).get(
                "summary"
            ),
            request_reasoning_effort=self._request_kwargs.get("reasoning", {}).get(
                "effort"
            ),
            response_reasoning_effort=None,
            request_service_tier=self._request_kwargs.get("service_tier"),
            response_service_tier=None,
        )

        self._complete_response_data = None
        self._output_text = ""

        self._cleanup_completed = False
        self._cleanup_lock = threading.Lock()

    def __del__(self):
        """Cleanup when object is garbage collected"""
        if hasattr(self, "_cleanup_completed") and not self._cleanup_completed:
            self._ensure_cleanup()

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
                self._handle_exception(exc_val)
            else:
                self._process_complete_response()
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
                self._handle_exception(exc_val)
            else:
                self._process_complete_response()
        finally:
            if hasattr(self.__wrapped__, "__aexit__"):
                suppress = bool(await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb))
        return suppress

    def close(self):
        try:
            self._ensure_cleanup()
        finally:
            if hasattr(self.__wrapped__, "close"):
                return self.__wrapped__.close()

    async def aclose(self):
        try:
            self._ensure_cleanup()
        finally:
            if hasattr(self.__wrapped__, "aclose"):
                return await self.__wrapped__.aclose()

    def __iter__(self):
        """Synchronous iterator"""
        return self

    def __next__(self):
        """Synchronous iteration"""
        try:
            chunk = self.__wrapped__.__next__()
        except StopIteration:
            self._process_complete_response()
            raise
        except Exception as e:
            self._handle_exception(e)
            raise
        else:
            self._process_chunk(chunk)
            return chunk

    def __aiter__(self):
        """Async iterator"""
        return self

    async def __anext__(self):
        """Async iteration"""
        try:
            chunk = await self.__wrapped__.__anext__()
        except StopAsyncIteration:
            self._process_complete_response()
            raise
        except Exception as e:
            self._handle_exception(e)
            raise
        else:
            self._process_chunk(chunk)
            return chunk

    def _process_chunk(self, chunk):
        """Process a streaming chunk"""
        if hasattr(chunk, "type"):
            if chunk.type == "response.output_text.delta":
                if hasattr(chunk, "delta") and chunk.delta:
                    self._output_text += chunk.delta
            elif chunk.type == "response.completed" and hasattr(chunk, "response"):
                self._complete_response_data = chunk.response

        if hasattr(chunk, "delta"):
            if hasattr(chunk.delta, "text") and chunk.delta.text:
                self._output_text += chunk.delta.text

        if hasattr(chunk, "response") and chunk.response:
            self._complete_response_data = chunk.response

    @dont_throw
    def _process_complete_response(self):
        """Process the complete response and emit span"""
        with self._cleanup_lock:
            if self._cleanup_completed:
                return

            try:
                if self._complete_response_data:
                    parsed_response = parse_response(self._complete_response_data)

                    self._traced_data.response_id = parsed_response.id
                    self._traced_data.response_model = parsed_response.model
                    self._traced_data.output_text = self._output_text

                    if parsed_response.usage:
                        self._traced_data.usage = parsed_response.usage

                    if parsed_response.output:
                        self._traced_data.output_blocks = {
                            block.id: block for block in parsed_response.output
                        }

                    responses[parsed_response.id] = self._traced_data

                set_data_attributes(self._traced_data, self._span)
                self._span.set_status(StatusCode.OK)
                self._span.end()
                self._cleanup_completed = True

            except Exception as e:
                if self._span and self._span.is_recording():
                    self._span.set_attribute(ERROR_TYPE, e.__class__.__name__)
                    self._span.set_status(StatusCode.ERROR, str(e))
                    self._span.end()
                self._cleanup_completed = True

    @dont_throw
    def _handle_exception(self, exception):
        """Handle exceptions during streaming"""
        with self._cleanup_lock:
            if self._cleanup_completed:
                return

            if self._span and self._span.is_recording():
                self._span.set_attribute(ERROR_TYPE, exception.__class__.__name__)
                self._span.record_exception(exception)
                self._span.set_status(StatusCode.ERROR, str(exception))
                self._span.end()

            self._cleanup_completed = True

    @dont_throw
    def _ensure_cleanup(self):
        """Ensure cleanup happens even if stream is not fully consumed"""
        with self._cleanup_lock:
            if self._cleanup_completed:
                return

            try:
                if self._span and self._span.is_recording():
                    set_data_attributes(self._traced_data, self._span)
                    self._span.set_status(StatusCode.OK)
                    self._span.end()

                self._cleanup_completed = True

            except Exception:
                self._cleanup_completed = True
