from dataclasses import asdict
from enum import Enum
from typing import Union

from opentelemetry._events import Event, EventLogger
from opentelemetry.semconv._incubating.attributes import \
    gen_ai_attributes as GenAIAttributes

from opentelemetry.instrumentation.writer.event_models import (ChoiceEvent,
                                                               MessageEvent)
from opentelemetry.instrumentation.writer.utils import (dont_throw,
                                                        model_as_dict,
                                                        should_emit_events,
                                                        should_send_prompts)


class Roles(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


VALID_MESSAGE_ROLES = {role.value for role in Roles}
"""The valid roles for naming the message event."""

EVENT_ATTRIBUTES = {
    # Should be GenAIAttributes.GenAiSystemValues.WRITER.value but it's not defined in the opentelemetry-semconv package
    GenAIAttributes.GEN_AI_SYSTEM: "writer"
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
def emit_choice_events(response, event_logger):
    response_dict = model_as_dict(response)

    for choice in response_dict.get("choices", []):
        message = choice.get("message")

        if message:
            emit_event(
                ChoiceEvent(
                    index=choice.get("index", 0),
                    message={
                        "content": message.get("content"),
                        "role": message.get("role", "assistant"),
                    },
                    finish_reason=choice.get("finish_reason"),
                ),
                event_logger=event_logger,
            )
        elif choice.get("text") is not None:
            emit_event(
                ChoiceEvent(
                    index=choice.get("index", 0),
                    message={
                        "content": choice.get("text"),
                        "role": "assistant",
                    },
                    finish_reason=choice.get("finish_reason"),
                ),
                event_logger=event_logger,
            )


@dont_throw
def emit_streaming_response_events(
    accumulated_content: str, finish_reason: Union[str, None], event_logger
):
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
        body.pop("role", None)
    else:
        name = "gen_ai.user.message"

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
