"""
Guardrails class for running guarded operations through the Traceloop client.
"""
import asyncio
import time
from typing import Callable, Awaitable, cast

import httpx
from opentelemetry.trace.status import Status, StatusCode

from traceloop.sdk.tracing import get_tracer
from traceloop.sdk.evaluator.evaluator import Evaluator

from .model import (
    GuardedOutput,
    Guard,
    OnFailureHandler,
    GuardedFunctionResult,
    GuardInput,
    FailureResult,
    GuardExecutionError,
)


class Guardrails:
    """
    Guardrails class for running guarded operations.

    Access via the Traceloop client:
        client = Traceloop.init(api_key="...")

    Usage:
        g = client.guardrails.create(
            guard=lambda z: z["score"] > 0.8,
            on_failure=OnFailure.raise_exception("Quality check failed"),
        )
        result1 = await g.run(agent1)
        result2 = await g.run(agent2)
    """

    _evaluator: Evaluator
    _async_http: httpx.AsyncClient
    _guard: Guard
    _on_failure: OnFailureHandler

    def __init__(self, async_http_client: httpx.AsyncClient):
        self._async_http = async_http_client
        self._evaluator = Evaluator(async_http_client)

    def create(
        self,
        guard: Guard,
        on_failure: OnFailureHandler,
    ) -> "Guardrails":
        """
        Create a new guardrail instance with the given guard and failure handler.

        Args:
            guard: Function that receives guard_input and returns bool.
                   True = pass, False = fail.
            on_failure: Called when guard returns False.

        Returns:
            Guardrails: A new instance configured with the given guard and on_failure.

        Example:
            g1 = client.guardrails.create(
                guard=lambda z: z["score"] > 0.8,
                on_failure=OnFailure.raise_exception("Quality check failed"),
            )
            g2 = client.guardrails.create(
                guard=lambda z: z["score"] > 0.5,
                on_failure=OnFailure.log("Warning"),
            )
            # g1 and g2 are independent instances
        """
        instance = Guardrails(self._async_http)
        instance._guard = guard
        instance._on_failure = on_failure
        return instance

    async def run(
        self,
        func_to_guard: Callable[
            [], Awaitable[GuardedOutput[GuardedFunctionResult, GuardInput]]
        ],
    ) -> GuardedFunctionResult | FailureResult:
        """
        Execute a function with guardrail protection.

        Must call create() first to configure guard and on_failure.

        Args:
            func_to_guard: Async function that returns GuardedOutput[T, Z].
                           Executed immediately inside run().

        Returns:
            T | F: The result from GuardedOutput.result, or the on_failure return value

        Raises:
            GuardValidationError: If guard returns False and on_failure raises
            GuardExecutionError: If the guard function throws an exception during execution
            ValueError: If create() was not called first

        Example:
            g = client.guardrails.create(
                guard=lambda z: z["score"] > 0.8,
                on_failure=OnFailure.raise_exception("Quality check failed"),
            )
            result = await g.run(generate_email)
        """
        if self._guard is None or self._on_failure is None:
            raise ValueError("Must call create() before run()")

        with get_tracer() as tracer:
            with tracer.start_as_current_span("guardrail.run") as span:
                start_time = time.perf_counter()

                # 1. Execute func_to_guard
                output: GuardedOutput[
                    GuardedFunctionResult, GuardInput
                ] = await func_to_guard()

                # 2. Run guard
                try:
                    guard_result = self._guard(output.guard_input)
                    if asyncio.iscoroutine(guard_result):
                        guard_result = await guard_result
                    guard_passed = bool(guard_result)
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise GuardExecutionError(
                        message="Guard execution failed",
                        original_exception=e,
                        guard_input=output.guard_input,
                    ) from e

                duration_ms = (time.perf_counter() - start_time) * 1000

                span.set_attribute("guardrail.passed", guard_passed)
                span.set_attribute("guardrail.duration_ms", duration_ms)

                # 3. Handle failure
                if not guard_passed:
                    failure_result = self._on_failure(output)
                    if asyncio.iscoroutine(failure_result):
                        failure_result = await failure_result
                    return cast(FailureResult, failure_result)

                # 4. Guard passed, return result
                return output.result
