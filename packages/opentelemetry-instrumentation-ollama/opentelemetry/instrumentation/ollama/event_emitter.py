from dataclasses import asdict
from enum import Enum
from typing import Dict, List, Union

from opentelemetry._events import Event
from opentelemetry.instrumentation.ollama.event_models import (
    ChoiceEvent,
    MessageEvent,
    ToolCall,
)
from opentelemetry.instrumentation.ollama.utils import (
    dont_throw,
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

EVENT_ATTRIBUTES = {GenAIAttributes.GEN_AI_SYSTEM: "ollama"}
"""The attributes to be used for the event."""


@dont_throw
def emit_message_events(llm_request_type, args, kwargs, event_logger):
    json_data = kwargs.get("json", {})
    if llm_request_type == LLMRequestTypeValues.CHAT:
        messages: List[Dict] = json_data.get("messages")
        for message in messages:
            content = message.get("content", {})
            images = message.get("images")
            if images is not None:
                content["images"] = images
            tool_calls = message.get("tool_calls", None)
            if tool_calls is not None:
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", ""),
                        function=tc.get("function"),
                        type="function",
                    )
                    for tc in tool_calls
                ]
                for tool_call in tool_calls:
                    tool_call["function"]["arguments"] = tool_call["function"].get(
                        "arguments", ""
                    )

            role = message.get("role")
            emit_event(
                MessageEvent(content=content, role=role, tool_calls=tool_calls),
                event_logger,
            )
    elif (
        llm_request_type == LLMRequestTypeValues.COMPLETION
        or LLMRequestTypeValues.EMBEDDING
    ):
        prompt = json_data.get("prompt", "")
        emit_event(MessageEvent(content=prompt, role="user"), event_logger)
    else:
        raise ValueError(
            "It wasn't possible to emit the input events due to an unknown llm_request_type."
        )


@dont_throw
def emit_choice_events(llm_request_type, response: dict, event_logger):
    if llm_request_type == LLMRequestTypeValues.CHAT:
        finish_reason = response.get("done_reason") or "unknown"
        emit_event(
            ChoiceEvent(
                index=0,
                message={
                    "content": response.get("message", {}).get("content"),
                    "role": response.get("message").get("role", "assistant"),
                },
                finish_reason=finish_reason,
            ),
            event_logger,
        )
    elif llm_request_type == LLMRequestTypeValues.COMPLETION:
        finish_reason = response.get("done_reason")
        emit_event(
            ChoiceEvent(
                index=0,
                message={"content": response.get("response"), "role": "assistant"},
                finish_reason=finish_reason or "unknown",
            ),
            event_logger,
        )
    elif llm_request_type == LLMRequestTypeValues.EMBEDDING:
        emit_event(
            ChoiceEvent(
                index=0,
                message={"content": response.get("embedding"), "role": "assistant"},
                finish_reason="unknown",
            ),
            event_logger,
        )
    else:
        raise ValueError(
            "It wasn't possible to emit the choice events due to an unknown llm_request_type."
        )


def emit_event(event: Union[MessageEvent, ChoiceEvent], event_logger) -> None:
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


def _emit_choice_event(event: ChoiceEvent, event_logger) -> None:
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
