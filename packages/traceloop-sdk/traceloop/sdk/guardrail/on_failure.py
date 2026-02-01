"""
Built-in failure handlers for guardrails.

On failure handlers are called when a guard returns False.
They receive the full GuardedResult and can raise, log, or perform custom actions.
"""
import logging
from typing import Any, Callable, Type

from .model import GuardedResult, GuardValidationError

logger = logging.getLogger("traceloop.guardrail")


class OnFailure:
    """Built-in failure handlers."""

    @staticmethod
    def raise_exception(
        message: str = "Guard validation failed",
        exception_type: Type[Exception] = GuardValidationError
    ) -> Callable[["GuardedResult[Any, Any]"], None]:
        """
        Raise an exception on failure.

        Args:
            message: Error message to include in the exception
            exception_type: The exception class to raise (default: GuardValidationError)

        Example:
            on_failure=OnFailure.raise_exception("PII detected in response")
            on_failure=OnFailure.raise_exception("Invalid input", exception_type=ValueError)
        """

        def handler(output: "GuardedResult[Any, Any]") -> None:
            if exception_type is GuardValidationError:
                raise exception_type(message, output)
            else:
                raise exception_type(message)

        return handler

    @staticmethod
    def log(
        level: int = logging.WARNING,
        message: str = "Guard validation failed",
    ) -> Callable[["GuardedResult[Any, Any]"], Any]:
        """
        Log failure and continue (return the result anyway).

        Args:
            level: Logging level (default: WARNING)
            message: Log message prefix

        Example:
            on_failure=OnFailure.log(message="Toxic content detected")
        """

        def handler(output: "GuardedResult[Any, Any]") -> Any:
            logger.log(level, f"{message}: guard_inputs_count={len(output.guard_inputs)}")
            return output.result

        return handler

    @staticmethod
    def noop() -> Callable[["GuardedResult[Any, Any]"], Any]:
        """
        Do nothing on failure (shadow mode).

        Useful for testing guards in production without affecting behavior.

        Example:
            on_failure=OnFailure.noop()  # Just observe, don't block
        """

        def handler(output: "GuardedResult[Any, Any]") -> Any:
            return output.result

        return handler

    @staticmethod
    def return_value(
        value: Any,
    ) -> Callable[["GuardedResult[Any, Any]"], Any]:
        """
        Return a fallback value on failure.

        Args:
            value: The value to return when guard fails

        Example:
            on_failure=OnFailure.return_value("Sorry, I can't help with that.")
            on_failure=OnFailure.return_value({"error": "blocked", "reason": "PII"})
        """

        def handler(output: "GuardedResult[Any, Any]") -> Any:
            return value

        return handler
