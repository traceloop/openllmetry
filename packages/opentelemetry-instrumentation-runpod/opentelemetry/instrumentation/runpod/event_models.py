from dataclasses import dataclass
from typing import Any, TypedDict


class CompletionMessage(TypedDict):
    """Represents a message produced by the instrumented call."""

    content: Any
    role: str


@dataclass
class MessageEvent:
    """Represents the payload submitted to the endpoint."""

    content: Any
    role: str = "user"


@dataclass
class ChoiceEvent:
    """Represents the response returned by the endpoint."""

    index: int
    message: CompletionMessage
    finish_reason: str = "unknown"
