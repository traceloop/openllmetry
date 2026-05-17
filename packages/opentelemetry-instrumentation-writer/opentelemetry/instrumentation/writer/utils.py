import logging
import os
import traceback
from importlib.metadata import version
from typing import Any

from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.trace import Span
from writerai.types import ChatCompletion, Completion
from writerai.types.chat_completion import ChatCompletionChoice
from writerai.types.chat_completion_message import ChatCompletionMessage
from writerai.types.shared.tool_call import Function, ToolCall

from opentelemetry import context as context_api
from opentelemetry.instrumentation.writer import Config

GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_SYSTEM_WRITER = "writer"

try:
    _PYDANTIC_VERSION = version("pydantic")
except ImportError:
    _PYDANTIC_VERSION = "0.0.0"

TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"


def set_span_attribute(span: Span, name: str, value: Any | None) -> None:
    if value is not None and value != "":
        span.set_attribute(name, value)


def should_send_prompts() -> bool:
    return (
        os.getenv(TRACELOOP_TRACE_CONTENT) or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.

    @param func: The function to wrap
    @return: The wrapper function
    """
    # Obtain a logger specific to the function's module
    logger = logging.getLogger(func.__module__)

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug(
                "OpenLLMetry failed to trace in %s, error: %s",
                func.__name__,
                traceback.format_exc(),
            )
            if Config.exception_logger:
                Config.exception_logger(e)

    return wrapper


@dont_throw
def error_metrics_attributes(exception) -> dict:
    return {
        GEN_AI_SYSTEM: GEN_AI_SYSTEM_WRITER,
        "error.type": exception.__class__.__name__,
        "error.message": str(exception),
    }


def request_type_by_method(method_name):
    if method_name == "chat":
        return LLMRequestTypeValues.CHAT
    elif method_name == "create":
        return LLMRequestTypeValues.COMPLETION
    else:
        return LLMRequestTypeValues.UNKNOWN


@dont_throw
def response_attributes(response, method) -> dict:
    response_dict = model_as_dict(response)

    return {
        GEN_AI_SYSTEM: GEN_AI_SYSTEM_WRITER,
        SpanAttributes.LLM_RESPONSE_MODEL: response_dict.get("model"),
        SpanAttributes.LLM_REQUEST_TYPE: request_type_by_method(method).value,
    }


def should_emit_events() -> bool:
    """
    Checks if the instrumentation isn't using the legacy attributes
    and if the event logger is not None.
    """

    return not Config.use_legacy_attributes


def model_as_dict(model) -> dict:
    if _PYDANTIC_VERSION < "2.0.0":
        return model.dict()
    if hasattr(model, "model_dump"):
        return model.model_dump()
    elif hasattr(model, "parse"):
        return model_as_dict(model.parse())
    else:
        return model


def initialize_accumulated_response(stream):
    request_url = stream.response.request.url.path

    if "chat" in request_url:
        return ChatCompletion(
            id="",
            choices=[],
            created=0,
            model="",
            object="chat.completion",
        )
    elif "completions" in request_url:
        return Completion(choices=[])
    else:
        raise ValueError(f"Unknown stream type. Request url: {request_url}")


def initialize_choice():
    return ChatCompletionChoice(
        index=0,
        finish_reason="stop",
        message=ChatCompletionMessage(
            content="",
            role="assistant",
            tool_calls=[],
        ),
    )


def initialize_tool_call():
    return ToolCall(
        id="", function=Function(name="", arguments=""), type="function", index=0
    )


def enhance_list_size(current_list, desired_size):
    if current_list is None:
        current_list = [None] * desired_size
    else:
        if desired_size < len(current_list):
            raise ValueError(
                f"Desired size ({desired_size} can't be less than actual size ({len(current_list)})."
            )
        current_list += [None] * (desired_size - len(current_list))
