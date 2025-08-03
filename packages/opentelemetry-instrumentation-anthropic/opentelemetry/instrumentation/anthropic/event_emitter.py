from dataclasses import asdict
from enum import Enum
import json
from typing import Optional, Union

from opentelemetry._events import Event, EventLogger
from opentelemetry.instrumentation.anthropic.event_models import (
    ChoiceEvent,
    MessageEvent,
    ToolCall,
)
from opentelemetry.instrumentation.anthropic.utils import (
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


class Roles(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


VALID_MESSAGE_ROLES = {role.value for role in Roles}
"""The valid roles for naming the message event."""

EVENT_ATTRIBUTES = {
    GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.ANTHROPIC.value
}
"""The attributes to be used for the event."""


def emit_input_events(event_logger: Optional[EventLogger], kwargs):
    if kwargs.get("prompt") is not None:
        emit_event(
            MessageEvent(content=kwargs.get("prompt"), role="user"), event_logger
        )

    elif kwargs.get("messages") is not None:
        if kwargs.get("system"):
            emit_event(
                MessageEvent(content=kwargs.get("system"), role="system"), event_logger
            )
        for message in kwargs.get("messages"):
            emit_event(
                MessageEvent(content=message.get("content"), role=message.get("role")),
                event_logger,
            )
    if kwargs.get("tools") is not None:
        emit_event(
            MessageEvent(content={"tools": kwargs.get("tools")}, role="user"),
            event_logger,
        )


def emit_response_events(event_logger: Optional[EventLogger], response):
    if not isinstance(response, dict):
        response = dict(response)

    if response.get("completion"):
        emit_event(
            ChoiceEvent(
                index=0,
                message={
                    "content": response.get("completion"),
                    "role": response.get("role", "assistant"),
                },
                finish_reason=response.get("stop_reason"),
            ),
            event_logger,
        )
    elif response.get("content"):
        for i, completion in enumerate(response.get("content")):
            # Parse message
            if completion.type == "text":
                message = {
                    "content": completion.text,
                    "role": response.get("role", "assistant"),
                }
            elif completion.type == "thinking":
                message = {
                    "content": completion.thinking,
                    "role": response.get("role", "assistant"),
                }
            elif completion.type == "tool_use":
                message = {
                    "content": None,
                    "role": response.get("role", "assistant"),
                }
            else:
                message = {
                    "content": None,
                    "role": response.get("role", "assistant"),
                }

            # Parse tool calls
            if completion.type == "tool_use":
                tool_calls = [
                    ToolCall(
                        id=completion.id,
                        function={
                            "name": completion.name,
                            "arguments": completion.input,
                        },
                        type="function",
                    )
                ]
            else:
                tool_calls = None

            # Emit the event
            emit_event(
                ChoiceEvent(
                    index=i,
                    message=message,
                    finish_reason=response.get("stop_reason"),
                    tool_calls=tool_calls,
                ),
                event_logger,
            )


def emit_streaming_response_events(
    event_logger: Optional[EventLogger], complete_response: dict
):
    for message in complete_response.get("events", []):
        # Parse tool calls
        if message.get("type") == "tool_use":
            tool_calls = [
                ToolCall(
                    id=message.get("id"),
                    function={
                        "name": message.get("name"),
                        "arguments": json.loads(message.get("input", '{}')),
                    },
                    type="function",
                )
            ]
            event = ChoiceEvent(
                index=message.get("index", 0),
                message={
                    "content": None,
                    "role": message.get("role", "assistant"),
                },
                finish_reason=message.get("finish_reason", "unknown"),
                tool_calls=tool_calls,
            )
        else:
            event = ChoiceEvent(
                index=message.get("index", 0),
                message={
                    "content": {
                        "type": message.get("type"),
                        "content": message.get("text"),
                    },
                    "role": message.get("role", "assistant"),
                },
                finish_reason=message.get("finish_reason", "unknown"),
            )
        emit_event(event, event_logger)


def emit_event(
    event: Union[MessageEvent, ChoiceEvent], event_logger: EventLogger
) -> None:
    """
    Emit an event to the OpenTelemetry SDK.

    Args:
        event: The event to emit.
    """
    if not should_emit_events():
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
