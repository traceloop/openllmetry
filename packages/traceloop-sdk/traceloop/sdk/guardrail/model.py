"""
Data models and type definitions for the guardrail system.
"""
from dataclasses import dataclass
from typing import TypeVar, Any, Union, Callable, Awaitable, Generic

# Type variables for generic typing
GuardInput = TypeVar("GuardInput")
GuardedFunctionResult = TypeVar("GuardedFunctionResult")
FailureResult = TypeVar("FailureResult")

# Type aliases for guards and handlers
Guard = Union[Callable[[Any], bool], Callable[[Any], Awaitable[bool]]]

# Type for input mapper: takes function result, returns list of guard inputs (one per guard)
InputMapper = Callable[[GuardedFunctionResult], list[GuardInput]]


@dataclass
class GuardedResult(Generic[GuardedFunctionResult, GuardInput]):
    """
    Container passed to on_failure handler with result and guard inputs.

    Attributes:
        result: The original value returned by the guarded function
        guard_inputs: List of inputs that were passed to each guard

    Example:
        def my_failure_handler(output: GuardedResult[str, dict]) -> str:
            # Access the original result
            original = output.result
            # Access what was checked
            inputs = output.guard_inputs
            return "Fallback response"
    """

    result: GuardedFunctionResult
    guard_inputs: list[GuardInput]


OnFailureHandler = Union[
    Callable[["GuardedResult[Any, Any]"], FailureResult],
    Callable[["GuardedResult[Any, Any]"], Awaitable[FailureResult]],
]


class GuardrailError(Exception):
    """
    Base exception for all guardrail-related errors.

    Use this to catch any guardrail error:
        try:
            result = await guardrail.run(my_func)
        except GuardrailError as e:
            # Handles GuardValidationError, GuardExecutionError, GuardInputTypeError
            pass
    """
    pass


class GuardValidationError(GuardrailError):
    """
    Raised when guard fails and on_failure handler raises.

    Attributes:
        message: Error description
        output: The GuardedResult containing result and guard inputs
    """

    def __init__(self, message: str, output: "GuardedResult[Any, Any]"):
        self.output = output
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.args[0]} (guard_inputs: {self.output.guard_inputs})"


class GuardExecutionError(GuardrailError):
    """
    Raised when the guard function itself throws an exception during execution.

    This is different from GuardValidationError which is raised when the guard
    returns False. GuardExecutionError indicates the guard could not complete.

    Attributes:
        message: Error description
        original_exception: The exception that was raised by the guard
        guard_input: The input that was passed to the guard
        guard_index: The index of the guard that failed (0-based)
    """

    def __init__(
        self,
        message: str,
        original_exception: Exception,
        guard_input: Any,
        guard_index: int | None = None,
    ):
        self.original_exception = original_exception
        self.guard_input = guard_input
        self.guard_index = guard_index
        super().__init__(message)

    def __str__(self) -> str:
        index_info = f" [guard {self.guard_index}]" if self.guard_index is not None else ""
        return (
            f"{self.args[0]}{index_info}: {self.original_exception} "
            f"(guard_input: {self.guard_input})"
        )


class GuardInputTypeError(GuardrailError):
    """
    Raised when a guard_input doesn't match the expected type of its guard function.

    This error occurs during input validation before guards are executed.

    Attributes:
        message: Error description
        guard_index: The index of the guard with type mismatch (0-based)
        expected_type: The type annotation from the guard function
        actual_type: The actual type of the guard_input
        validation_error: The underlying Pydantic validation error
    """

    def __init__(
        self,
        message: str,
        guard_index: int,
        expected_type: Any,
        actual_type: type,
        validation_error: Exception | None = None,
    ):
        self.guard_index = guard_index
        self.expected_type = expected_type
        self.actual_type = actual_type
        self.validation_error = validation_error
        super().__init__(message)

    def __str__(self) -> str:
        return (
            f"{self.args[0]} [guard {self.guard_index}]: "
            f"expected {self.expected_type}, got {self.actual_type.__name__}"
        )
