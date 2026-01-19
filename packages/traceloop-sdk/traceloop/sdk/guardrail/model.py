"""
Data models for the guardrail system.
"""
from dataclasses import dataclass
from typing import TypeVar, Generic, Any

T = TypeVar("T")  # Result type (returned to caller)
Z = TypeVar("Z")  # Guard input type (passed to guard for evaluation)


@dataclass
class GuardedOutput(Generic[T, Z]):
    """
    Container that separates what to return from what to evaluate.

    Attributes:
        result: The value returned to the caller when guard passes
        guard_input: The value passed to the guard for evaluation

    Example:
        async def generate_email() -> GuardedOutput[str, dict]:
            text = await llm.complete("Write a customer email...")
            return GuardedOutput(
                result=text,
                guard_input={"text": text},
            )
    """

    result: T
    guard_input: Z


class GuardValidationError(Exception):
    """
    Raised when guard fails and on_failure handler raises.

    Attributes:
        message: Error description
        output: The full GuardedOutput that failed validation
    """

    def __init__(self, message: str, output: GuardedOutput[Any, Any]):
        self.output = output
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.args[0]} (guard_input: {self.output.guard_input})"
