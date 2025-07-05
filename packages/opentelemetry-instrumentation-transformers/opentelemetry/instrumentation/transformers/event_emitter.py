from enum import Enum
from typing import Union

from opentelemetry._events import Event
from opentelemetry.instrumentation.transformers.event_models import (
    CompletionEvent,
    PromptEvent,
)
from opentelemetry.instrumentation.transformers.utils import (
    dont_throw,
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

EVENT_ATTRIBUTES = {GenAIAttributes.GEN_AI_SYSTEM: "transformers"}
"""The attributes to be used for the event."""


@dont_throw
def emit_prompt_events(args, kwargs, event_logger) -> None:
    if not should_emit_events() or event_logger is None:
        return

    if args and len(args) > 0:
        prompts_list = args[0]
    else:
        prompts_list = kwargs.get("args")

    if isinstance(prompts_list, str):
        prompts_list = [prompts_list]

    for prompt in prompts_list:
        emit_event(PromptEvent(content=prompt, role="user"), event_logger)


@dont_throw
def emit_response_events(response, event_logger) -> None:
    if response and len(response) > 0:
        for i, completion in enumerate(response):
            emit_event(
                CompletionEvent(
                    index=i,
                    message={
                        "content": completion.get("generated_text"),
                        "role": "assistant",
                    },
                    finish_reason="unknown",
                ),
                event_logger,
            )


def emit_event(event: Union[PromptEvent, CompletionEvent], event_logger) -> None:
    """
    Emit an event to the OpenTelemetry SDK.

    Args:
        event: The event to emit.
    """
    if not should_emit_events():
        return

    if isinstance(event, PromptEvent):
        _emit_prompt_event(event, event_logger)
    elif isinstance(event, CompletionEvent):
        _emit_completion_event(event, event_logger)
    else:
        raise TypeError("Unsupported event type")


def _emit_prompt_event(event: PromptEvent, event_logger) -> None:
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


def _emit_completion_event(event: CompletionEvent, event_logger) -> None:
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
