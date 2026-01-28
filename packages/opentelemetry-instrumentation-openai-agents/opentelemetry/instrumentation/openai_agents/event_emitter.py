from dataclasses import asdict
from enum import Enum
from typing import Union

from opentelemetry._events import Event
from opentelemetry.instrumentation.openai_agents.event_models import (
    ChoiceEvent,
    MessageEvent,
    ToolStartEvent,
    ToolEndEvent,
)
from opentelemetry.instrumentation.openai_agents.utils import (
    should_emit_events,
    should_send_prompts,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from .config import Config


class Roles(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


VALID_MESSAGE_ROLES = {role.value for role in Roles}
"""The valid roles for naming the message event."""

EVENT_ATTRIBUTES = {
    GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.OPENAI.value
}
"""The attributes to be used for the event."""


def emit_event(
    event: Union[MessageEvent, ChoiceEvent, ToolStartEvent, ToolEndEvent]
) -> None:
    """
    Emit an event to the OpenTelemetry SDK.

    Args:
        event: The event to emit.
    """
    if not should_emit_events():
        return

    if isinstance(event, MessageEvent):
        _emit_message_event(event)
    elif isinstance(event, ChoiceEvent):
        _emit_choice_event(event)
    elif isinstance(event, ToolStartEvent):
        _emit_tool_start_event(event)
    elif isinstance(event, ToolEndEvent):
        _emit_tool_end_event(event)
    else:
        raise TypeError("Unsupported event type")


def _emit_message_event(event: MessageEvent) -> None:
    body = asdict(event)

    if event.role in VALID_MESSAGE_ROLES:
        name = "gen_ai.{}.message".format(event.role)
        # According to the semantic conventions, the role is conditionally required
        # if available and not equal to the "role" in the message name. So, remove
        # the role from the body if it is the same as in the event name.
        body.pop("role", None)
    else:
        # Unknown role: default to user message event but keep the original role
        # in the body per semantic conventions (role is required when it differs
        # from the role implied by the event name).
        name = "gen_ai.user.message"
        # Role is intentionally kept in body since it differs from "user"

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

    Config.event_logger.emit(Event(name=name, body=body, attributes=EVENT_ATTRIBUTES))


def _emit_choice_event(event: ChoiceEvent) -> None:
    body = asdict(event)
    if event.message["role"] == Roles.ASSISTANT.value:
        # According to the semantic conventions, the role is conditionally required
        # if available and not equal to "assistant", so remove the role from the
        # body if it is "assistant".
        body["message"].pop("role", None)

    if event.tool_calls is None:
        del body["tool_calls"]

    if not should_send_prompts():
        body["message"].pop("content", None)
        if body.get("tool_calls") is not None:
            for tool_call in body["tool_calls"]:
                tool_call["function"].pop("arguments", None)

    # Follow langchain pattern: extract content to gen_ai.output.message and remove
    # message key. Only set gen_ai.output.message if content exists (it may have
    # been removed by should_send_prompts check)
    if "content" in body["message"]:
        body["gen_ai.output.message"] = body["message"]["content"]
    del body["message"]

    Config.event_logger.emit(
        Event(name="gen_ai.choice", body=body, attributes=EVENT_ATTRIBUTES)
    )


def _emit_tool_start_event(event: ToolStartEvent) -> None:
    body = asdict(event)
    Config.event_logger.emit(
        Event(name="gen_ai.tool.start", body=body, attributes=EVENT_ATTRIBUTES)
    )


def _emit_tool_end_event(event: ToolEndEvent) -> None:
    body = asdict(event)
    Config.event_logger.emit(
        Event(name="gen_ai.tool.end", body=body, attributes=EVENT_ATTRIBUTES)
    )
