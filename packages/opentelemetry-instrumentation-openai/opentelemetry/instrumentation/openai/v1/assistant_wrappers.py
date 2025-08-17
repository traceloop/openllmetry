import logging
import time

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.instrumentation.openai.shared import (
    _set_span_attribute,
    model_as_dict,
)
from opentelemetry.instrumentation.openai.shared.config import Config
from opentelemetry.instrumentation.openai.shared.event_emitter import emit_event
from opentelemetry.instrumentation.openai.shared.event_models import (
    ChoiceEvent,
    MessageEvent,
)
from opentelemetry.instrumentation.openai.utils import (
    _with_tracer_wrapper,
    dont_throw,
    should_emit_events,
)
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode

from openai._legacy_response import LegacyAPIResponse
from openai.types.beta.threads.run import Run

logger = logging.getLogger(__name__)

assistants = {}
runs = {}


@_with_tracer_wrapper
def assistants_create_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    response = wrapped(*args, **kwargs)

    assistants[response.id] = {
        "model": kwargs.get("model"),
        "instructions": kwargs.get("instructions"),
    }

    return response


@_with_tracer_wrapper
def runs_create_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    thread_id = kwargs.get("thread_id")
    instructions = kwargs.get("instructions")

    try:
        response = wrapped(*args, **kwargs)
        response_dict = model_as_dict(response)

        runs[thread_id] = {
            "start_time": time.time_ns(),
            "assistant_id": kwargs.get("assistant_id"),
            "instructions": instructions,
            "run_id": response_dict.get("id"),
        }

        return response
    except Exception as e:
        runs[thread_id] = {
            "exception": e,
            "end_time": time.time_ns(),
        }
        raise


@_with_tracer_wrapper
def runs_retrieve_wrapper(tracer, wrapped, instance, args, kwargs):
    @dont_throw
    def process_response(response):
        if type(response) is LegacyAPIResponse:
            parsed_response = response.parse()
        else:
            parsed_response = response
        assert type(parsed_response) is Run

        if parsed_response.thread_id in runs:
            thread_id = parsed_response.thread_id
            runs[thread_id]["end_time"] = time.time_ns()
            if parsed_response.usage:
                runs[thread_id]["usage"] = parsed_response.usage

    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    try:
        response = wrapped(*args, **kwargs)
        process_response(response)
        return response
    except Exception as e:
        thread_id = kwargs.get("thread_id")
        if thread_id in runs:
            runs[thread_id]["exception"] = e
            runs[thread_id]["end_time"] = time.time_ns()
        raise


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

    # Use the span as current context to ensure events get proper trace context
    with trace.use_span(span, end_on_exit=False):
        if exception := run.get("exception"):
            span.set_attribute(ERROR_TYPE, exception.__class__.__name__)
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(exception)))
            span.end()
            return response

        prompt_index = 0
        if assistants.get(run["assistant_id"]) is not None or Config.enrich_assistant:
            if Config.enrich_assistant:
                assistant = model_as_dict(
                    instance._client.beta.assistants.retrieve(run["assistant_id"])
                )
                assistants[run["assistant_id"]] = assistant
            else:
                assistant = assistants[run["assistant_id"]]

            _set_span_attribute(
                span,
                SpanAttributes.LLM_SYSTEM,
                "openai",
            )
            _set_span_attribute(
                span,
                SpanAttributes.LLM_REQUEST_MODEL,
                assistant["model"],
            )
            _set_span_attribute(
                span,
                SpanAttributes.LLM_RESPONSE_MODEL,
                assistant["model"],
            )
            if should_emit_events():
                emit_event(MessageEvent(content=assistant["instructions"], role="system"))
            else:
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.role", "system"
                )
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.content",
                    assistant["instructions"],
                )
            prompt_index += 1
        _set_span_attribute(
            span, f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.role", "system"
        )
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.content",
            run["instructions"],
        )
        if should_emit_events():
            emit_event(MessageEvent(content=run["instructions"], role="system"))
        prompt_index += 1

        completion_index = 0
        for msg in messages:
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{completion_index}"
            content = msg.get("content")

            message_content = content[0].get("text").get("value")
            message_role = msg.get("role")
            if message_role in ["user", "system"]:
                if should_emit_events():
                    emit_event(MessageEvent(content=message_content, role=message_role))
                else:
                    _set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.role",
                        message_role,
                    )
                    _set_span_attribute(
                        span,
                        f"{SpanAttributes.LLM_PROMPTS}.{prompt_index}.content",
                        message_content,
                    )
                prompt_index += 1
            else:
                if should_emit_events():
                    emit_event(
                        ChoiceEvent(
                            index=completion_index,
                            message={"content": message_content, "role": message_role},
                        )
                    )
                else:
                    _set_span_attribute(span, f"{prefix}.role", msg.get("role"))
                    _set_span_attribute(span, f"{prefix}.content", message_content)
                    _set_span_attribute(
                        span, f"gen_ai.response.{completion_index}.id", msg.get("id")
                    )
                completion_index += 1

        if run.get("usage"):
            usage_dict = model_as_dict(run.get("usage"))
            _set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
                usage_dict.get("completion_tokens"),
            )
            _set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
                usage_dict.get("prompt_tokens"),
            )

    span.end(run.get("end_time"))

    return response


@_with_tracer_wrapper
def runs_create_and_stream_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    assistant_id = kwargs.get("assistant_id")
    instructions = kwargs.get("instructions")

    span = tracer.start_span(
        "openai.assistant.run_stream",
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value},
    )

    # Use the span as current context to ensure events get proper trace context
    with trace.use_span(span, end_on_exit=False):
        i = 0
        if assistants.get(assistant_id) is not None or Config.enrich_assistant:
            if Config.enrich_assistant:
                assistant = model_as_dict(
                    instance._client.beta.assistants.retrieve(assistant_id)
                )
                assistants[assistant_id] = assistant
            else:
                assistant = assistants[assistant_id]

            _set_span_attribute(
                span, SpanAttributes.LLM_REQUEST_MODEL, assistants[assistant_id]["model"]
            )
            _set_span_attribute(
                span,
                SpanAttributes.LLM_SYSTEM,
                "openai",
            )
            _set_span_attribute(
                span,
                SpanAttributes.LLM_RESPONSE_MODEL,
                assistants[assistant_id]["model"],
            )
            if should_emit_events():
                emit_event(
                    MessageEvent(
                        content=assistants[assistant_id]["instructions"], role="system"
                    )
                )
            else:
                _set_span_attribute(
                    span, f"{SpanAttributes.LLM_PROMPTS}.{i}.role", "system"
                )
                _set_span_attribute(
                    span,
                    f"{SpanAttributes.LLM_PROMPTS}.{i}.content",
                    assistants[assistant_id]["instructions"],
                )
            i += 1
        if should_emit_events():
            emit_event(MessageEvent(content=instructions, role="system"))
        else:
            _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.{i}.role", "system")
            _set_span_attribute(
                span, f"{SpanAttributes.LLM_PROMPTS}.{i}.content", instructions
            )

        from opentelemetry.instrumentation.openai.v1.event_handler_wrapper import (
            EventHandleWrapper,
        )

        kwargs["event_handler"] = EventHandleWrapper(
            original_handler=kwargs["event_handler"],
            span=span,
        )

        try:
            response = wrapped(*args, **kwargs)
            return response
        except Exception as e:
            span.set_attribute(ERROR_TYPE, e.__class__.__name__)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.end()
            raise
