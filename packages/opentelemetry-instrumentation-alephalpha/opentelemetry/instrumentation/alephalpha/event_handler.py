from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Literal, Optional, TypedDict, Union

from opentelemetry._events import Event
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

EVENT_ATTRIBUTES = {GenAIAttributes.GEN_AI_SYSTEM: "alephalpha"}
"""The attributes to be used for the event."""


class _FunctionToolCall(TypedDict):
    function_name: str
    arguments: Optional[dict[str, Any]]


class ToolCall(TypedDict):
    """Represents a tool call in the AI model."""

    id: str
    function: _FunctionToolCall
    type: Literal["function"]


class CompletionMessage(TypedDict):
    """Represents a message in the AI model."""

    content: Any
    role: str = "assistant"


@dataclass
class MessageEvent:
    """Represents an input event for the AI model."""

    content: Any
    role: str = "user"
    tool_calls: Optional[List[ToolCall]] = None
    llm_request_model: Optional[str] = None


@dataclass
class ChoiceEvent:
    """Represents a completion event for the AI model."""

    index: int
    message: CompletionMessage
    input_tokens: int
    output_tokens: int
    finish_reason: str = "unknown"
    tool_calls: Optional[List[ToolCall]] = None

    @property
    def total_tokens(self) -> Optional[int]:
        """Returns the total number of tokens used in the event."""
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


def emit_event(event: Union[MessageEvent, ChoiceEvent], event_logger) -> None:
    from opentelemetry.instrumentation.alephalpha import (
        should_emit_events,
    )

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


def _emit_message_event(event: MessageEvent, event_logger) -> None:
    from opentelemetry.instrumentation.alephalpha import (
        should_send_prompts,
    )

    body = {
        "content": event.content,
        "role": event.role,
        "tool_calls": event.tool_calls,
    }

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


def _emit_choice_event(event: ChoiceEvent, event_logger) -> None:
    from opentelemetry.instrumentation.alephalpha import (
        should_send_prompts,
    )

    body = {
        "index": event.index,
        "message": event.message,
        "finish_reason": event.finish_reason,
        "tool_calls": event.tool_calls,
    }

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
