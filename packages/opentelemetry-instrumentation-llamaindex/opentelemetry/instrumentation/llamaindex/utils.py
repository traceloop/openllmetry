import dataclasses
import json
import logging
import os
import traceback
from contextlib import asynccontextmanager

from opentelemetry import context as context_api
from opentelemetry._events import Event
from opentelemetry.instrumentation.llamaindex.config import Config
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)
EVENT_ATTRIBUTES = {GenAIAttributes.GEN_AI_SYSTEM: "llamaindex"}


def _with_tracer_wrapper(func):
    def _with_tracer(tracer):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


@asynccontextmanager
async def start_as_current_span_async(tracer, *args, **kwargs):
    with tracer.start_as_current_span(*args, **kwargs) as span:
        yield span


def should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
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


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        elif hasattr(o, "json"):
            return o.json()
        elif hasattr(o, "to_json"):
            return o.to_json()
        return super().default(o)


@dont_throw
def process_request(span, args, kwargs):
    if should_send_prompts():
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_INPUT,
            json.dumps({"args": args, "kwargs": kwargs}, cls=JSONEncoder),
        )


@dont_throw
def process_response(span, res):
    if should_send_prompts():
        span.set_attribute(
            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
            json.dumps(res, cls=JSONEncoder),
        )


def is_role_valid(role: str) -> bool:
    return role in ["user", "assistant", "system", "tool"]


def is_content_enabled() -> bool:
    capture_content = os.environ.get(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false"
    )

    return capture_content.lower() == "true"


def emit_message_event(*, content, role: str):
    body = {}

    if is_content_enabled():
        body["content"] = content
        body["role"] = role

    if is_role_valid(role):
        event_name = "gen_ai.{}.message".format(role)
    else:
        event_name = "gen_ai.{}.message".format("user")

    Config.event_logger.emit(Event(event_name, body=body, attributes=EVENT_ATTRIBUTES))


def emit_choice_event(
    *,
    index: int = 0,
    content,
    role: str,
    finish_reason: str,
):
    body = {"index": index, "finish_reason": finish_reason, "message": {}}

    if is_content_enabled():
        body["message"]["content"] = content
        body["message"]["role"] = role

    Config.event_logger.emit(
        Event("gen_ai.choice", body=body, attributes=EVENT_ATTRIBUTES)
    )
