from dataclasses import asdict
from enum import Enum
from typing import Union

from opentelemetry._events import Event, EventLogger
from opentelemetry.instrumentation.together.event_models import (
    CompletionEvent,
    PromptEvent,
)
from opentelemetry.instrumentation.together.utils import (
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

from together.types.chat_completions import ChatCompletionResponse
from together.types.completions import CompletionResponse


class Roles(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


VALID_MESSAGE_ROLES = {role.value for role in Roles}
"""The valid roles for naming the message event."""

EVENT_ATTRIBUTES = {GenAIAttributes.GEN_AI_SYSTEM: "together"}
"""The attributes to be used for the event."""


@dont_throw
def emit_prompt_events(event_logger, llm_request_type, kwargs):
    if llm_request_type == LLMRequestTypeValues.CHAT:
        for message in kwargs.get("messages"):
            emit_event(
                PromptEvent(
                    content=message.get("content"), role=message.get("role") or "user"
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
def emit_completion_event(
    event_logger,
    llm_request_type,
    response: Union[ChatCompletionResponse, CompletionResponse],
):
    if llm_request_type == LLMRequestTypeValues.CHAT:
        response: ChatCompletionResponse
        for choice in response.choices:
            emit_event(
                CompletionEvent(
                    index=choice.index,
                    message={
                        "content": choice.message.content,
                        "role": choice.message.role,
                    },
                    finish_reason=choice.finish_reason,
                ),
                event_logger,
            )
    elif llm_request_type == LLMRequestTypeValues.COMPLETION:
        response: CompletionResponse
        for choice in response.choices:
            emit_event(
                CompletionEvent(
                    index=choice.index,
                    message={"content": choice.text, "role": "assistant"},
                    finish_reason=choice.finish_reason,
                ),
                event_logger,
            )
    else:
        raise ValueError(
            "It wasn't possible to emit the choice events due to an unknown llm_request_type."
        )


def emit_event(
    event: Union[PromptEvent, CompletionEvent], event_logger: Union[EventLogger, None]
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


def _emit_message_event(event: PromptEvent, event_logger: EventLogger) -> None:
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


def _emit_choice_event(event: CompletionEvent, event_logger: EventLogger) -> None:
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
