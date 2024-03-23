import json
import logging
import time
from opentelemetry import context as context_api
from opentelemetry.instrumentation.openai.shared import (
    _set_span_attribute,
    model_as_dict,
)
from opentelemetry.trace import SpanKind
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY

from opentelemetry.semconv.ai import SpanAttributes, LLMRequestTypeValues

from opentelemetry.instrumentation.openai.utils import _with_tracer_wrapper

logger = logging.getLogger(__name__)

assistants = {}
runs = {}


@_with_tracer_wrapper
def assistants_create_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    response = wrapped(*args, **kwargs)

    assistants[response.id] = {"model": kwargs.get("model")}

    return response


@_with_tracer_wrapper
def runs_create_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    thread_id = kwargs.get("thread_id")

    response = wrapped(*args, **kwargs)

    runs[thread_id] = {
        "start_time": time.time_ns(),
        "assistant_id": kwargs.get("assistant_id"),
    }

    return response


@_with_tracer_wrapper
def runs_retrieve_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    thread_id = kwargs.get("thread_id")

    response = wrapped(*args, **kwargs)

    if response.id in runs:
        runs[thread_id]["end_time"] = time.time_ns()

    return response


@_with_tracer_wrapper
def messages_list_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    id = kwargs.get("thread_id")

    response = wrapped(*args, **kwargs)

    response_dict = model_as_dict(response)
    if id not in runs:
        return response

    run = runs[id]
    messages = sorted(response_dict["data"], key=lambda x: x["created_at"])

    span = tracer.start_span(
        "openai.assistant.run",
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value},
        start_time=run.get("start_time"),
    )

    _set_span_attribute(
        span, SpanAttributes.LLM_REQUEST_MODEL, assistants[run["assistant_id"]]["model"]
    )
    _set_span_attribute(
        span,
        SpanAttributes.LLM_RESPONSE_MODEL,
        assistants[run["assistant_id"]]["model"],
    )

    for i, msg in enumerate(messages):
        prefix = f"{SpanAttributes.LLM_PROMPTS}.{i}"
        content = json.dumps(msg.get("content"))

        _set_span_attribute(span, f"{prefix}.role", msg.get("role"))
        _set_span_attribute(span, f"{prefix}.content", content)

    span.end(run.get("end_time"))

    return response
