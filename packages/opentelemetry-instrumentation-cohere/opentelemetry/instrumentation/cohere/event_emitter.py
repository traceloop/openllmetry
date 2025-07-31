from dataclasses import asdict
from enum import Enum
from typing import Union

from opentelemetry._events import Event, EventLogger
from opentelemetry.instrumentation.cohere.event_models import ChoiceEvent, MessageEvent
from opentelemetry.instrumentation.cohere.utils import (
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

EVENT_ATTRIBUTES = {
    GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.COHERE.value
}
"""The attributes to be used for the event."""


def emit_input_event(event_logger, llm_request_type: str, kwargs):
    if not should_emit_events() or event_logger is None:
        return
    event_params = {}

    if llm_request_type == LLMRequestTypeValues.CHAT:
        event_params = {"content": kwargs.get("message"), "role": "user"}
    elif llm_request_type == LLMRequestTypeValues.RERANK:
        event_params = {
            "content": {
                "query": kwargs.get("query"),
                "documents": kwargs.get("documents"),
            },
            "role": "user",
        }
    elif llm_request_type == LLMRequestTypeValues.COMPLETION:
        event_params = {"content": kwargs.get("prompt"), "role": "user"}

    emit_event(MessageEvent(**event_params), event_logger)


def emit_response_events(event_logger, llm_request_type: str, response):
    if not should_emit_events() or event_logger is None:
        return

    if llm_request_type == LLMRequestTypeValues.COMPLETION:
        for index, generation in enumerate(response.generations):
            emit_event(
                _parse_response_event(index, llm_request_type, generation),
                event_logger,
            )
    else:
        emit_event(_parse_response_event(0, llm_request_type, response), event_logger)


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


def _parse_response_event(index: int, llm_request_type: str, response) -> ChoiceEvent:
    event_params = {"index": index, "finish_reason": "unknown"}

    if llm_request_type == LLMRequestTypeValues.RERANK:
        event_params["message"] = {
            "content": [
                {
                    "index": result.index,
                    "document": result.document,
                    "relevance_score": result.relevance_score,
                }
                for result in response.results
            ],
            "role": "assistant",
        }
    elif (
        llm_request_type == LLMRequestTypeValues.CHAT or LLMRequestTypeValues.COMPLETION
    ):
        event_params["message"] = {"content": response.text, "role": "assistant"}
        event_params["finish_reason"] = response.finish_reason

    return ChoiceEvent(**event_params)
