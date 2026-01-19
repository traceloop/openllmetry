"""
Data models and type definitions for the guardrail system.
"""
from dataclasses import dataclass
from typing import TypeVar, Generic, Any, Union, Callable, Awaitable

GuardedFunctionResult = TypeVar("GuardedFunctionResult")
GuardInput = TypeVar("GuardInput")
FailureResult = TypeVar("FailureResult")

# Type aliases for guards and handlers
Guard = Union[Callable[[GuardInput], bool], Callable[[GuardInput], Awaitable[bool]]]
OnFailureHandler = Union[
    Callable[
        ["GuardedFunctionOutput[GuardedFunctionResult, GuardInput]"], FailureResult
    ],
    Callable[
        ["GuardedFunctionOutput[GuardedFunctionResult, GuardInput]"],
        Awaitable[FailureResult],
    ],
]


@dataclass
class GuardedFunctionOutput(Generic[GuardedFunctionResult, GuardInput]):
    """
    Container that separates what to return from what to evaluate.

    Attributes:
        result: The original value returned by the guarded function
        guard_input: The value passed to the guard for evaluation

    Example:
        async def generate_email() -> GuardedFunctionOutput[str, dict]:
            text = await llm.complete("Write a customer email...")
            return GuardedFunctionOutput(
                result=text,
                guard_input={"text": text},
            )
    """

    result: GuardedFunctionResult
    guard_input: GuardInput


class GuardValidationError(Exception):
    """
    Raised when guard fails and on_failure handler raises.

    Attributes:
        message: Error description
        output: The full GuardedOutput that failed validation
    """

    def __init__(self, message: str, output: GuardedFunctionOutput[Any, Any]):
        self.output = output
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.args[0]} (guard_input: {self.output.guard_input})"


class GuardExecutionError(Exception):
    """
    Raised when the guard function itself throws an exception during execution.

    This is different from GuardValidationError which is raised when the guard
    returns False. GuardExecutionError indicates the guard could not complete.

    Attributes:
        message: Error description
        original_exception: The exception that was raised by the guard
        guard_input: The input that was passed to the guard
    """

    def __init__(
        self,
        message: str,
        original_exception: Exception,
        guard_input: Any,
    ):
        self.original_exception = original_exception
        self.guard_input = guard_input
        super().__init__(message)

    def __str__(self) -> str:
        return (
            f"{self.args[0]}: {self.original_exception} "
            f"(guard_input: {self.guard_input})"
        )
