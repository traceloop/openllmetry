from dataclasses import asdict
from enum import Enum
from typing import Union

from opentelemetry._events import Event, EventLogger
from opentelemetry.instrumentation.groq.event_models import ChoiceEvent, MessageEvent
from opentelemetry.instrumentation.groq.utils import (
    dont_throw,
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from groq.types.chat.chat_completion import ChatCompletion


class Roles(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


VALID_MESSAGE_ROLES = {role.value for role in Roles}
"""The valid roles for naming the message event."""

EVENT_ATTRIBUTES = {
    # Should be GenAIAttributes.GenAiSystemValues.GROQ.value but it's not defined in the opentelemetry-semconv package
    GenAIAttributes.GEN_AI_SYSTEM: "groq"
}
"""The attributes to be used for the event."""


@dont_throw
def emit_message_events(kwargs: dict, event_logger):
    for message in kwargs.get("messages", []):
        emit_event(
            MessageEvent(
                content=message.get("content"), role=message.get("role", "unknown")
            ),
            event_logger=event_logger,
        )


@dont_throw
def emit_choice_events(response: ChatCompletion, event_logger):
    for choice in response.choices:
        emit_event(
            ChoiceEvent(
                index=choice.index,
                message={
                    "content": choice.message.content,
                    "role": choice.message.role or "unknown",
                },
                finish_reason=choice.finish_reason,
            ),
            event_logger=event_logger,
        )


@dont_throw
def emit_streaming_response_events(
    accumulated_content: str, finish_reason: Union[str, None], event_logger
):
    """Emit events for streaming response."""
    emit_event(
        ChoiceEvent(
            index=0,
            message={"content": accumulated_content, "role": "assistant"},
            finish_reason=finish_reason or "unknown",
        ),
        event_logger,
    )


def emit_event(
    event: Union[MessageEvent, ChoiceEvent], event_logger: Union[EventLogger, None]
) -> None:
    """
    Emit an event to the OpenTelemetry SDK.

    Args:
        event: The event to emit.
    """
    if not should_emit_events() or event_logger is None:
        return

    if isinstance(event, MessageEvent):
        _emit_message_event(event, event_logger)
    elif isinstance(event, ChoiceEvent):
        _emit_choice_event(event, event_logger)
    else:
        raise TypeError("Unsupported event type")


def _emit_message_event(event: MessageEvent, event_logger: EventLogger) -> None:
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

    if not should_send_prompts():
        del body["content"]
        if body.get("tool_calls") is not None:
            for tool_call in body["tool_calls"]:
                tool_call["function"].pop("arguments", None)

    event_logger.emit(Event(name=name, body=body, attributes=EVENT_ATTRIBUTES))


def _emit_choice_event(event: ChoiceEvent, event_logger: EventLogger) -> None:
    body = asdict(event)
    if event.message["role"] == Roles.ASSISTANT.value:
        # According to the semantic conventions, the role is conditionally required if available
        # and not equal to "assistant", so remove the role from the body if it is "assistant".
        body["message"].pop("role", None)

    if event.tool_calls is None:
        del body["tool_calls"]

    if not should_send_prompts():
        body["message"].pop("content", None)
        if body.get("tool_calls") is not None:
            for tool_call in body["tool_calls"]:
                tool_call["function"].pop("arguments", None)

    event_logger.emit(
        Event(name="gen_ai.choice", body=body, attributes=EVENT_ATTRIBUTES)
    )
