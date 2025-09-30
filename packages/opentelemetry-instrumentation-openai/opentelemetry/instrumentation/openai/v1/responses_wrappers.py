import json
import logging
import pydantic
import re
import threading
import time

from openai import AsyncStream, Stream

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
    from typing import Any, Dict, List, Union

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

from openai._legacy_response import LegacyAPIResponse
from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_COMPLETION,
    GEN_AI_PROMPT,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_RESPONSE_ID,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
)
from opentelemetry.trace import SpanKind, Span, Status, StatusCode, Tracer
from opentelemetry import trace
from typing import Any, Optional, Union
from typing_extensions import NotRequired
from wrapt import ObjectProxy

from opentelemetry.instrumentation.openai.shared import (
    _set_span_attribute,
    model_as_dict,
    metric_shared_attributes,
    _get_openai_base_url,
)

from opentelemetry.instrumentation.openai.utils import (
    _with_responses_telemetry_wrapper,
    _with_tracer_wrapper,
    dont_throw,
    should_send_prompts,
)

SPAN_NAME = "openai.response"

logger = logging.getLogger(__name__)


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


responses: dict[str, TracedData] = {}


def parse_response(response: Union[LegacyAPIResponse, Response]) -> Response:
    if isinstance(response, LegacyAPIResponse):
        return response.parse()
    return response


def get_tools_from_kwargs(kwargs: dict) -> list[ToolParam]:
    tools_input = kwargs.get("tools", [])
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


class ResponseStream(ObjectProxy):
    """
    Stream wrapper for OpenAI Responses API streaming responses.
    Handles span lifecycle and data accumulation for streaming responses.
    Aligned with ChatStream pattern for consistency.
    """
    _span = None
    _instance = None
    _token_counter = None
    _choice_counter = None
    _duration_histogram = None
    _streaming_time_to_first_token = None
    _streaming_time_to_generate = None
    _start_time = None
    _request_kwargs = None
    _traced_data = None

    def __init__(
        self,
        span,
        response,
        traced_data,
        instance=None,
        token_counter=None,
        choice_counter=None,
        duration_histogram=None,
        streaming_time_to_first_token=None,
        streaming_time_to_generate=None,
        start_time=None,
        request_kwargs=None,
    ):
        super().__init__(response)
        self._span = span
        self._instance = instance
        self._traced_data = traced_data
        self._token_counter = token_counter
        self._choice_counter = choice_counter
        self._duration_histogram = duration_histogram
        self._streaming_time_to_first_token = streaming_time_to_first_token
        self._streaming_time_to_generate = streaming_time_to_generate
        self._start_time = start_time
        self._request_kwargs = request_kwargs or {}

        self._first_token = True
        self._time_of_first_token = self._start_time
        self._cleanup_lock = threading.Lock()
        self._cleanup_completed = False
        self._error_recorded = False

        # Store initial data in global responses dict
        if self._traced_data.response_id:
            responses[self._traced_data.response_id] = self._traced_data

    def __del__(self):
        """Cleanup when garbage collected."""
        self._ensure_cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Call the wrapped stream's __exit__
        result = self.__wrapped__.__exit__(exc_type, exc_val, exc_tb)

        # Perform cleanup
        try:
            self._ensure_cleanup()
        except Exception as cleanup_exception:
            logger.debug(
                "Error during ResponseStream cleanup in __exit__: %s", cleanup_exception
            )

        return result

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        result = await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb)

        # Perform cleanup
        try:
            self._ensure_cleanup()
        except Exception as cleanup_exception:
            logger.debug(
                "Error during ResponseStream cleanup in __aexit__: %s", cleanup_exception
            )

        return result

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = self.__wrapped__.__next__()
        except Exception as e:
            if isinstance(e, StopIteration):
                self._process_complete_response()
            else:
                if self._span and self._span.is_recording():
                    self._span.record_exception(e)
                    self._span.set_status(Status(StatusCode.ERROR, str(e)))
                    self._error_recorded = True
                self._ensure_cleanup()
            raise
        else:
            self._process_chunk(chunk)
            return chunk

    async def __anext__(self):
        try:
            chunk = await self.__wrapped__.__anext__()
        except Exception as e:
            if isinstance(e, StopAsyncIteration):
                self._process_complete_response()
            else:
                if self._span and self._span.is_recording():
                    self._span.record_exception(e)
                    self._span.set_status(Status(StatusCode.ERROR, str(e)))
                    self._error_recorded = True
                self._ensure_cleanup()
            raise
        else:
            self._process_chunk(chunk)
            return chunk

    def _process_chunk(self, chunk):
        """Process a streaming chunk and update TracedData."""
        if self._span and self._span.is_recording():
            self._span.add_event(name=f"{SpanAttributes.LLM_CONTENT_COMPLETION_CHUNK}")

        if self._first_token and self._streaming_time_to_first_token:
            self._time_of_first_token = time.time()
            self._streaming_time_to_first_token.record(
                self._time_of_first_token - self._start_time,
                attributes=self._shared_attributes()
            )
            self._first_token = False

        try:
            parsed_chunk = parse_response(chunk)

            # Update response_id if it becomes available
            if parsed_chunk.id and not self._traced_data.response_id:
                self._traced_data.response_id = parsed_chunk.id
                responses[parsed_chunk.id] = self._traced_data

            # Update TracedData with new information from chunk
            if hasattr(parsed_chunk, 'output'):
                # Merge output blocks
                new_blocks = {block.id: block for block in parsed_chunk.output}
                if self._traced_data.output_blocks is None:
                    self._traced_data.output_blocks = {}
                self._traced_data.output_blocks.update(new_blocks)

            if hasattr(parsed_chunk, 'usage') and parsed_chunk.usage:
                self._traced_data.usage = parsed_chunk.usage

            if hasattr(parsed_chunk, 'model') and parsed_chunk.model:
                self._traced_data.response_model = parsed_chunk.model

            # Update output_text if available
            if hasattr(parsed_chunk, 'output_text'):
                self._traced_data.output_text = parsed_chunk.output_text
            else:
                # Try to extract text from output blocks
                try:
                    if parsed_chunk.output and len(parsed_chunk.output) > 0:
                        first_output = parsed_chunk.output[0]
                        if hasattr(first_output, 'content') and first_output.content:
                            if len(first_output.content) > 0:
                                first_content = first_output.content[0]
                                if hasattr(first_content, 'text'):
                                    self._traced_data.output_text = first_content.text
                except Exception:
                    pass

            # Update global dict with latest data
            if self._traced_data.response_id:
                responses[self._traced_data.response_id] = self._traced_data

        except Exception as e:
            logger.debug("Error processing response chunk: %s", e)

    def _shared_attributes(self):
        """Get shared attributes for metrics."""
        return metric_shared_attributes(
            response_model=self._traced_data.response_model
            or self._traced_data.request_model
            or None,
            operation="response",
            server_address=_get_openai_base_url(self._instance),
            is_streaming=True,
        )

    @dont_throw
    def _process_complete_response(self):
        """Process the complete response and close the span."""
        if self._span and self._span.is_recording():
            set_data_attributes(self._traced_data, self._span)

        if self._token_counter and self._traced_data.usage:
            usage = self._traced_data.usage
            shared_attrs = self._shared_attributes()
            if hasattr(usage, 'input_tokens') and usage.input_tokens:
                attributes_with_token_type = {
                    **shared_attrs,
                    SpanAttributes.LLM_TOKEN_TYPE: "input",
                }
                self._token_counter.record(usage.input_tokens, attributes=attributes_with_token_type)
            if hasattr(usage, 'output_tokens') and usage.output_tokens:
                attributes_with_token_type = {
                    **shared_attrs,
                    SpanAttributes.LLM_TOKEN_TYPE: "output",
                }
                self._token_counter.record(usage.output_tokens, attributes=attributes_with_token_type)

        if self._choice_counter and self._traced_data.output_blocks:
            shared_attrs = self._shared_attributes()
            num_blocks = len(self._traced_data.output_blocks)
            if num_blocks > 0:
                self._choice_counter.add(num_blocks, attributes=shared_attrs)

        if self._duration_histogram and self._start_time:
            duration = time.time() - self._start_time
            self._duration_histogram.record(duration, attributes=self._shared_attributes())

        if self._streaming_time_to_generate and self._time_of_first_token:
            self._streaming_time_to_generate.record(
                time.time() - self._time_of_first_token,
                attributes=self._shared_attributes()
            )

        if self._span and self._span.is_recording():
            if not self._error_recorded:
                self._span.set_status(Status(StatusCode.OK))
            self._span.end()

        self._cleanup_completed = True
        logger.debug("ResponseStream span closed successfully")

    @dont_throw
    def _ensure_cleanup(self):
        """Ensure proper cleanup of streaming response."""
        with self._cleanup_lock:
            if self._cleanup_completed:
                logger.debug("ResponseStream cleanup already completed, skipping")
                return

            try:
                logger.debug("Starting ResponseStream cleanup")

                if self._span and self._span.is_recording():
                    if not self._error_recorded:
                        self._span.set_status(Status(StatusCode.OK))
                    self._span.end()
                    logger.debug("ResponseStream span closed in cleanup")

                self._cleanup_completed = True
                logger.debug("ResponseStream cleanup completed successfully")

            except Exception as e:
                logger.debug("Error during ResponseStream cleanup: %s", str(e))

                try:
                    if self._span and self._span.is_recording():
                        if not self._error_recorded:
                            self._span.set_status(Status(StatusCode.ERROR, "Cleanup failed"))
                        self._span.end()
                    self._cleanup_completed = True
                except Exception:
                    self._cleanup_completed = True


@dont_throw
def set_data_attributes(traced_response: TracedData, span: Span):
    _set_span_attribute(span, GEN_AI_SYSTEM, "openai")
    _set_span_attribute(span, GEN_AI_REQUEST_MODEL, traced_response.request_model)
    _set_span_attribute(span, GEN_AI_RESPONSE_ID, traced_response.response_id)
    _set_span_attribute(span, GEN_AI_RESPONSE_MODEL, traced_response.response_model)
    if usage := traced_response.usage:
        _set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, usage.input_tokens)
        _set_span_attribute(span, GEN_AI_USAGE_OUTPUT_TOKENS, usage.output_tokens)
        _set_span_attribute(
            span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total_tokens
        )
        if usage.input_tokens_details:
            _set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
                usage.input_tokens_details.cached_tokens,
            )

        # Usage - count of reasoning tokens
        reasoning_tokens = None
        # Support both dict-style and object-style `usage`
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

    # Reasoning attributes
    # Request - reasoning summary
    _set_span_attribute(
        span,
        f"{SpanAttributes.LLM_REQUEST_REASONING_SUMMARY}",
        traced_response.request_reasoning_summary or (),
    )
    # Request - reasoning effort
    _set_span_attribute(
        span,
        f"{SpanAttributes.LLM_REQUEST_REASONING_EFFORT}",
        traced_response.request_reasoning_effort or (),
    )
    # Response - reasoning effort
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
                f"{GEN_AI_PROMPT}.{prompt_index}.content",
                traced_response.instructions,
            )
            _set_span_attribute(span, f"{GEN_AI_PROMPT}.{prompt_index}.role", "system")
            prompt_index += 1

        if isinstance(traced_response.input, str):
            _set_span_attribute(
                span, f"{GEN_AI_PROMPT}.{prompt_index}.content", traced_response.input
            )
            _set_span_attribute(span, f"{GEN_AI_PROMPT}.{prompt_index}.role", "user")
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
                        f"{GEN_AI_PROMPT}.{prompt_index}.content",
                        stringified_content,
                    )
                    _set_span_attribute(
                        span,
                        f"{GEN_AI_PROMPT}.{prompt_index}.role",
                        block_dict.get("role"),
                    )
                    prompt_index += 1
                elif block_dict.get("type") == "computer_call_output":
                    _set_span_attribute(
                        span, f"{GEN_AI_PROMPT}.{prompt_index}.role", "computer-call"
                    )
                    output_image_url = block_dict.get("output", {}).get("image_url")
                    if output_image_url:
                        _set_span_attribute(
                            span,
                            f"{GEN_AI_PROMPT}.{prompt_index}.content",
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
                        span, f"{GEN_AI_PROMPT}.{prompt_index}.role", "assistant"
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
                        f"{GEN_AI_PROMPT}.{prompt_index}.content",
                        json.dumps(call_content),
                    )
                    prompt_index += 1
                # TODO: handle other block types

        _set_span_attribute(span, f"{GEN_AI_COMPLETION}.0.role", "assistant")
        if traced_response.output_text:
            _set_span_attribute(
                span, f"{GEN_AI_COMPLETION}.0.content", traced_response.output_text
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
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.id",
                    block_dict.get("id"),
                )
                _set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.name",
                    block_dict.get("name"),
                )
                _set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.arguments",
                    block_dict.get("arguments"),
                )
                tool_call_index += 1
            elif block_dict.get("type") == "file_search_call":
                _set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.id",
                    block_dict.get("id"),
                )
                _set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.name",
                    "file_search_call",
                )
                tool_call_index += 1
            elif block_dict.get("type") == "web_search_call":
                _set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.id",
                    block_dict.get("id"),
                )
                _set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.name",
                    "web_search_call",
                )
                tool_call_index += 1
            elif block_dict.get("type") == "computer_call":
                _set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.id",
                    block_dict.get("call_id"),
                )
                _set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.name",
                    "computer_call",
                )
                _set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.arguments",
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
                        span, f"{GEN_AI_COMPLETION}.0.reasoning", reasoning_value
                    )
            # TODO: handle other block types, in particular other calls


@dont_throw
@_with_responses_telemetry_wrapper
def responses_get_or_create_wrapper(
    tracer: Tracer,
    token_counter,
    choice_counter,
    duration_histogram,
    exception_counter,
    streaming_time_to_first_token,
    streaming_time_to_generate,
    wrapped,
    instance,
    args,
    kwargs,
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = tracer.start_span(
        SPAN_NAME,
        kind=SpanKind.CLIENT,
    )

    with trace.use_span(span, end_on_exit=False):
        try:
            start_time = time.time()
            response = wrapped(*args, **kwargs)
            end_time = time.time()
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0

            if duration > 0 and duration_histogram:
                duration_histogram.record(duration, attributes={"error.type": e.__class__.__name__})
            if exception_counter:
                exception_counter.add(1, attributes={"error.type": e.__class__.__name__})

            span.set_attribute(ERROR_TYPE, e.__class__.__name__)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.end()

            raise

        if isinstance(response, Stream):
            response_id = kwargs.get("response_id")
            existing_data = {}
            if response_id and response_id in responses:
                existing_data = responses[response_id].model_dump()

            traced_data = TracedData(
                start_time=time.time_ns(),  # Use nanoseconds for TracedData
                response_id=response_id or "",
                input=process_input(
                    kwargs.get("input", existing_data.get("input", []))
                ),
                instructions=kwargs.get(
                    "instructions", existing_data.get("instructions")
                ),
                tools=get_tools_from_kwargs(kwargs) or existing_data.get("tools", []),
                output_blocks=existing_data.get("output_blocks", {}),
                usage=existing_data.get("usage"),
                output_text=kwargs.get(
                    "output_text", existing_data.get("output_text", "")
                ),
                request_model=kwargs.get(
                    "model", existing_data.get("request_model", "")
                ),
                response_model=existing_data.get("response_model", ""),
                request_reasoning_summary=(
                    kwargs.get("reasoning", {}).get(
                        "summary", existing_data.get("request_reasoning_summary")
                    )
                ),
                request_reasoning_effort=(
                    kwargs.get("reasoning", {}).get(
                        "effort", existing_data.get("request_reasoning_effort")
                    )
                ),
                response_reasoning_effort=kwargs.get("reasoning", {}).get("effort"),
            )

            return ResponseStream(
                span,
                response,
                traced_data,
                instance,
                token_counter,
                choice_counter,
                duration_histogram,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                start_time,
                kwargs,
            )
    parsed_response = parse_response(response)

    existing_data = responses.get(parsed_response.id)
    if existing_data is None:
        existing_data = {}
    else:
        existing_data = existing_data.model_dump()

    request_tools = get_tools_from_kwargs(kwargs)

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
            input=process_input(existing_data.get("input", kwargs.get("input"))),
            instructions=existing_data.get("instructions", kwargs.get("instructions")),
            tools=merged_tools if merged_tools else None,
            output_blocks={block.id: block for block in parsed_response.output}
            | existing_data.get("output_blocks", {}),
            usage=existing_data.get("usage", parsed_response.usage),
            output_text=existing_data.get("output_text", parsed_response_output_text),
            request_model=existing_data.get("request_model", kwargs.get("model")),
            response_model=existing_data.get("response_model", parsed_response.model),
            # Reasoning attributes
            request_reasoning_summary=(
                kwargs.get("reasoning", {}).get(
                    "summary", existing_data.get("request_reasoning_summary")
                )
            ),
            request_reasoning_effort=(
                kwargs.get("reasoning", {}).get(
                    "effort", existing_data.get("request_reasoning_effort")
                )
            ),
            response_reasoning_effort=kwargs.get("reasoning", {}).get("effort"),
        )
        responses[parsed_response.id] = traced_data
    except Exception:
        return response

    if parsed_response.status == "completed":
        span = tracer.start_span(
            SPAN_NAME,
            kind=SpanKind.CLIENT,
            start_time=int(traced_data.start_time),
        )
        set_data_attributes(traced_data, span)
        span.end()

    return response


@dont_throw
@_with_responses_telemetry_wrapper
async def async_responses_get_or_create_wrapper(
    tracer: Tracer,
    token_counter,
    choice_counter,
    duration_histogram,
    exception_counter,
    streaming_time_to_first_token,
    streaming_time_to_generate,
    wrapped,
    instance,
    args,
    kwargs,
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    span = tracer.start_span(
        SPAN_NAME,
        kind=SpanKind.CLIENT,
    )

    with trace.use_span(span, end_on_exit=False):
        try:
            start_time = time.time()
            response = await wrapped(*args, **kwargs)
            end_time = time.time()
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time if "start_time" in locals() else 0

            if duration > 0 and duration_histogram:
                duration_histogram.record(duration, attributes={"error.type": e.__class__.__name__})
            if exception_counter:
                exception_counter.add(1, attributes={"error.type": e.__class__.__name__})

            span.set_attribute(ERROR_TYPE, e.__class__.__name__)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.end()

            raise

        if isinstance(response, (Stream, AsyncStream)):
            response_id = kwargs.get("response_id")
            existing_data = {}
            if response_id and response_id in responses:
                existing_data = responses[response_id].model_dump()

            traced_data = TracedData(
                start_time=time.time_ns(),  # Use nanoseconds for TracedData
                response_id=response_id or "",
                input=process_input(
                    kwargs.get("input", existing_data.get("input", []))
                ),
                instructions=kwargs.get(
                    "instructions", existing_data.get("instructions", "")
                ),
                tools=get_tools_from_kwargs(kwargs) or existing_data.get("tools", []),
                output_blocks=existing_data.get("output_blocks", {}),
                usage=existing_data.get("usage"),
                output_text=kwargs.get("output_text", existing_data.get("output_text")),
                request_model=kwargs.get("model", existing_data.get("request_model")),
                response_model=existing_data.get("response_model"),
                request_reasoning_summary=(
                    kwargs.get("reasoning", {}).get(
                        "summary", existing_data.get("request_reasoning_summary")
                    )
                ),
                request_reasoning_effort=(
                    kwargs.get("reasoning", {}).get(
                        "effort", existing_data.get("request_reasoning_effort")
                    )
                ),
                response_reasoning_effort=kwargs.get("reasoning", {}).get("effort"),
            )

            return ResponseStream(
                span,
                response,
                traced_data,
                instance,
                token_counter,
                choice_counter,
                duration_histogram,
                streaming_time_to_first_token,
                streaming_time_to_generate,
                start_time,
                kwargs,
            )
    parsed_response = parse_response(response)

    existing_data = responses.get(parsed_response.id)
    if existing_data is None:
        existing_data = {}
    else:
        existing_data = existing_data.model_dump()

    request_tools = get_tools_from_kwargs(kwargs)

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
            input=process_input(existing_data.get("input", kwargs.get("input"))),
            instructions=existing_data.get("instructions", kwargs.get("instructions")),
            tools=merged_tools if merged_tools else None,
            output_blocks={block.id: block for block in parsed_response.output}
            | existing_data.get("output_blocks", {}),
            usage=existing_data.get("usage", parsed_response.usage),
            output_text=existing_data.get("output_text", parsed_response_output_text),
            request_model=existing_data.get("request_model", kwargs.get("model")),
            response_model=existing_data.get("response_model", parsed_response.model),
            # Reasoning attributes
            request_reasoning_summary=(
                kwargs.get("reasoning", {}).get(
                    "summary", existing_data.get("request_reasoning_summary")
                )
            ),
            request_reasoning_effort=(
                kwargs.get("reasoning", {}).get(
                    "effort", existing_data.get("request_reasoning_effort")
                )
            ),
            response_reasoning_effort=kwargs.get("reasoning", {}).get("effort"),
        )
        responses[parsed_response.id] = traced_data
    except Exception:
        return response

    if parsed_response.status == "completed":
        span = tracer.start_span(
            SPAN_NAME,
            kind=SpanKind.CLIENT,
            start_time=int(traced_data.start_time),
        )
        set_data_attributes(traced_data, span)
        span.end()

    return response


@dont_throw
@_with_tracer_wrapper
def responses_cancel_wrapper(tracer: Tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    response = wrapped(*args, **kwargs)
    if isinstance(response, Stream):
        return response
    parsed_response = parse_response(response)
    existing_data = responses.pop(parsed_response.id, None)
    if existing_data is not None:
        span = tracer.start_span(
            SPAN_NAME,
            kind=SpanKind.CLIENT,
            start_time=existing_data.start_time,
            record_exception=True,
        )
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

    response = await wrapped(*args, **kwargs)
    if isinstance(response, (Stream, AsyncStream)):
        return response
    parsed_response = parse_response(response)
    existing_data = responses.pop(parsed_response.id, None)
    if existing_data is not None:
        span = tracer.start_span(
            SPAN_NAME,
            kind=SpanKind.CLIENT,
            start_time=existing_data.start_time,
            record_exception=True,
        )
        span.record_exception(Exception("Response cancelled"))
        set_data_attributes(existing_data, span)
        span.end()
    return response
