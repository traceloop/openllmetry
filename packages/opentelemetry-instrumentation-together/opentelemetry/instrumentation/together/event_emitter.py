from dataclasses import asdict
from enum import Enum
from typing import Union

from opentelemetry._logs import Logger, LogRecord
from opentelemetry.instrumentation.together.event_models import (
    CompletionEvent,
    PromptEvent,
)
from opentelemetry.instrumentation.together.span_utils import map_finish_reason
from opentelemetry.instrumentation.together.utils import (
    TOGETHER_PROVIDER,
    dont_throw,
    enum_value,
    model_as_dict,
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
)


class Roles(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


VALID_MESSAGE_ROLES = {role.value for role in Roles}
"""The valid roles for naming the message event."""

EVENT_ATTRIBUTES = {GenAIAttributes.GEN_AI_PROVIDER_NAME: TOGETHER_PROVIDER}
"""The attributes to be used for the event."""


def _tool_calls(tool_calls):
    """Tool calls (dicts or SDK models) as OpenAI-format dicts, or None if there are none."""
    return [model_as_dict(tool_call) for tool_call in tool_calls or []] or None


@dont_throw
def emit_prompt_events(event_logger, llm_request_type, kwargs):
    if llm_request_type == LLMRequestTypeValues.CHAT:
        for message in kwargs.get("messages") or []:
            message = model_as_dict(message)
            emit_event(
                PromptEvent(
                    content=message.get("content"),
                    role=enum_value(message.get("role")) or "user",
                    tool_calls=_tool_calls(message.get("tool_calls")),
                    id=message.get("tool_call_id"),
                ),
                event_logger,
            )
    elif llm_request_type == LLMRequestTypeValues.COMPLETION:
        emit_event(PromptEvent(content=kwargs.get("prompt"), role="user"), event_logger)
    else:
        raise ValueError(
            "It wasn't possible to emit the input events due to an unknown llm_request_type."
        )


@dont_throw
def emit_completion_event(event_logger, llm_request_type, response):
    """`response` is the response model as a dict (see `model_as_dict`), so the
    response class, which lives in a different module in each together major
    version, is never imported."""
    if llm_request_type not in (LLMRequestTypeValues.CHAT, LLMRequestTypeValues.COMPLETION):
        raise ValueError(
            "It wasn't possible to emit the choice events due to an unknown llm_request_type."
        )

    for choice in response.get("choices") or []:
        choice = model_as_dict(choice)
        if llm_request_type == LLMRequestTypeValues.CHAT:
            message = choice.get("message") or {}
            content = message.get("content")
            role = enum_value(message.get("role")) or Roles.ASSISTANT.value
            tool_calls = _tool_calls(message.get("tool_calls"))
        else:
            content = choice.get("text")
            role = Roles.ASSISTANT.value
            tool_calls = None
        emit_event(
            CompletionEvent(
                index=choice.get("index"),
                message={"content": content, "role": role},
                finish_reason=map_finish_reason(choice.get("finish_reason")),
                tool_calls=tool_calls,
            ),
            event_logger,
        )


def emit_event(
    event: Union[PromptEvent, CompletionEvent], event_logger: Union[Logger, None]
) -> None:
    """
    Emit an event to the OpenTelemetry SDK.

    Args:
        event: The event to emit.
    """
    if not should_emit_events():
        return

    if isinstance(event, PromptEvent):
        _emit_message_event(event, event_logger)
    elif isinstance(event, CompletionEvent):
        _emit_choice_event(event, event_logger)
    else:
        raise TypeError("Unsupported event type")


def _emit_message_event(event: PromptEvent, event_logger: Logger) -> None:
    body = asdict(event)

    if event.role in VALID_MESSAGE_ROLES:
        name = "gen_ai.{}.message".format(event.role)
        # According to the semantic conventions, the role is conditionally required if available
        # and not equal to the "role" in the message name. So, remove the role from the body if
        # it is the same as the in the event name.
        body.pop("role", None)
    else:
        name = "gen_ai.user.message"

    # According to the semantic conventions, only the assistant role has tool call
    if event.role != Roles.ASSISTANT.value and event.tool_calls is not None:
        del body["tool_calls"]
    elif event.tool_calls is None:
        del body["tool_calls"]

    # Only tool messages carry the id of the tool call they answer.
    if event.id is None:
        del body["id"]

    if not should_send_prompts():
        del body["content"]
        _strip_tool_call_arguments(body)

    log_record = LogRecord(
        body=body,
        attributes=EVENT_ATTRIBUTES,
        event_name=name
    )
    event_logger.emit(log_record)


def _emit_choice_event(event: CompletionEvent, event_logger: Logger) -> None:
    body = asdict(event)
    if event.message["role"] == Roles.ASSISTANT.value:
        # According to the semantic conventions, the role is conditionally required if available
        # and not equal to "assistant", so remove the role from the body if it is "assistant".
        body["message"].pop("role", None)

    if event.tool_calls is None:
        del body["tool_calls"]

    if not should_send_prompts():
        body["message"].pop("content", None)
        _strip_tool_call_arguments(body)

    log_record = LogRecord(
        body=body,
        attributes=EVENT_ATTRIBUTES,
        event_name="gen_ai.choice"

    )
    event_logger.emit(log_record)


def _strip_tool_call_arguments(body):
    """Tool call arguments are content, so they are dropped when content tracing is off.

    `body` comes from `dataclasses.asdict`, which copies nested dicts, so the app's
    own messages are not modified."""
    for tool_call in body.get("tool_calls") or []:
        (tool_call.get("function") or {}).pop("arguments", None)
