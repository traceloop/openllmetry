from dataclasses import asdict
from enum import Enum
from typing import Union

from opentelemetry._logs import Logger, LogRecord
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
def emit_message_events(kwargs: dict, event_logger) -> None:
    messages = kwargs.get("messages", [])

    if messages:
        for message in kwargs.get("messages", []):
            emit_event(
                MessageEvent(
                    content=message.get("content"),
                    role=message.get("role", "unknown"),
                    tool_calls=message.get("tool_calls", []),
                ),
                event_logger=event_logger,
            )

    elif prompt := kwargs.get("prompt"):
        emit_event(
            MessageEvent(content=prompt, role="user"),
            event_logger=event_logger,
        )


@dont_throw
def emit_choice_events(response, event_logger) -> None:
    response_dict = model_as_dict(response)

    for choice in response_dict.get("choices", []):
        message = choice.get("message")

        if message:
            emit_event(
                ChoiceEvent(
                    index=choice.get("index", 0),
                    message=MessageEvent(
                        content=message.get("content"),
                        role=message.get("role", "assistant"),
                        tool_calls=message.get("tool_calls") or [],
                    ),
                    finish_reason=choice.get("finish_reason"),
                ),
                event_logger=event_logger,
            )
        elif choice.get("text") is not None:
            emit_event(
                ChoiceEvent(
                    index=choice.get("index", 0),
                    message=MessageEvent(
                        content=choice.get("text"),
                        role="assistant",
                    ),
                    finish_reason=choice.get("finish_reason", "unknown"),
                ),
                event_logger=event_logger,
            )


def emit_event(
    event: Union[MessageEvent, ChoiceEvent], event_logger: Union[Logger, None]
) -> None:
    if not should_emit_events() or event_logger is None:
        return

    if isinstance(event, MessageEvent):
        _emit_message_event(event, event_logger)
    elif isinstance(event, ChoiceEvent):
        _emit_choice_event(event, event_logger)
    else:
        raise TypeError("Unsupported event type")


def _emit_message_event(event: MessageEvent, event_logger: Logger) -> None:
    body = asdict(event)

    if event.role in VALID_MESSAGE_ROLES:
        name = f"gen_ai.{event.role}.message"
        body.pop("role", None)
    else:
        name = "gen_ai.user.message"

    if event.role != Roles.ASSISTANT.value:
        body.pop("tool_calls", None)

    if not should_send_prompts():
        del body["content"]
        if body.get("tool_calls") is not None:
            for tool_call in body["tool_calls"]:
                tool_call["function"].pop("arguments", None)

    log_record = LogRecord(
        body=body,
        attributes=EVENT_ATTRIBUTES,
        event_name=name
    )
    event_logger.emit(log_record)


def _emit_choice_event(event: ChoiceEvent, event_logger: Logger) -> None:
    body = asdict(event)
    if event.message.role == Roles.ASSISTANT.value:
        body["message"].pop("role", None)

    if event.message.tool_calls is None:
        del body["message"]["tool_calls"]

    if not should_send_prompts():
        body["message"].pop("content", None)
        if body["message"].get("tool_calls") is not None:
            for tool_call in body["message"]["tool_calls"]:
                tool_call["function"].pop("arguments", None)

    log_record = LogRecord(
        body=body,
        attributes=EVENT_ATTRIBUTES,
        event_name="gen_ai.choice"

    )
    event_logger.emit(log_record)
