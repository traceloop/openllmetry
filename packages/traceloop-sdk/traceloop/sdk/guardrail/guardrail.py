"""
Guardrails class for running guarded operations through the Traceloop client.
"""
import asyncio
import inspect
import time
from typing import Callable, Awaitable, cast, Optional

import httpx
from pydantic import TypeAdapter
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
    GuardInputTypeError,
)


class Guardrails:
    """
    Guardrails class for running guarded operations.

    Access via the Traceloop client:
        client = Traceloop.init(api_key="...")

    Usage:
        g = client.guardrails.create(
            guards=[
                lambda z: z["score"] > 0.8,
                EvaluatorMadeByTraceloop.pii_detector().as_guard(...),
            ],
            on_failure=OnFailure.raise_exception("Guard failed"),
        )
        result = await g.run(my_function)
    """

    _evaluator: Evaluator
    _async_http: httpx.AsyncClient
    _guards: list[Guard]
    _on_failure: OnFailureHandler
    _run_all: bool
    _parallel: bool

    def __init__(self, async_http_client: httpx.AsyncClient):
        self._async_http = async_http_client
        self._evaluator = Evaluator(async_http_client)
        self._guards = []
        self._on_failure = None
        self._run_all = False
        self._parallel = True

    def create(
        self,
        guards: list[Guard],
        on_failure: OnFailureHandler,
        run_all: bool = False,
        parallel: bool = True,
    ) -> "Guardrails":
        """
        Create a new guardrail instance with the given guards and failure handler.

        Args:
            guards: List of guard functions. Each receives its corresponding
                    guard_input and returns bool. True = pass, False = fail.
            on_failure: Called when any guard returns False.
            run_all: If True, run all guards before handling failures.
                     If False (default), stop at first failure.
            parallel: If True (default), run guards in parallel.
                      If False, run guards sequentially.

        Returns:
            Guardrails: A new instance configured with the given guards.

        Example:
            g = client.guardrails.create(
                guards=[
                    lambda z: z["score"] > 0.8,
                    EvaluatorMadeByTraceloop.pii_detector().as_guard(...),
                ],
                on_failure=OnFailure.raise_exception("Guard failed"),
                parallel=True,
            )
        """
        instance = Guardrails(self._async_http)
        instance._guards = guards
        instance._on_failure = on_failure
        instance._run_all = run_all
        instance._parallel = parallel
        return instance

    def _validate_inputs(self, guard_inputs: list[GuardInput]) -> None:
        """
        Validate guard_inputs match guards in count and type.

        Uses Pydantic TypeAdapter for robust type validation including
        Pydantic models, TypedDicts, generic types, etc.

        Raises:
            ValueError: If the number of guard_inputs doesn't match the number of guards
            GuardInputTypeError: If a guard_input doesn't match the guard's expected type
        """
        # Length validation
        if len(guard_inputs) != len(self._guards):
            raise ValueError(
                f"Number of guard_inputs ({len(guard_inputs)}) "
                f"must match number of guards ({len(self._guards)})"
            )

        # Type validation
        for i, (guard, guard_input) in enumerate(zip(self._guards, guard_inputs)):
            try:
                signature = inspect.signature(guard)
            except (ValueError, TypeError):
                # Can't get signature (e.g., built-in functions)
                continue

            params = list(signature.parameters.values())
            if not params:
                continue

            first_param = params[0]
            expected_type = first_param.annotation

            # Skip if no type annotation (e.g., lambdas)
            if expected_type is inspect.Parameter.empty:
                continue

            try:
                TypeAdapter(expected_type).validate_python(guard_input)
            except Exception as e:
                raise GuardInputTypeError(
                    message=f"Guard {i} expected {expected_type}, but got {type(guard_input).__name__}",
                    guard_index=i,
                    expected_type=expected_type,
                    actual_type=type(guard_input),
                    validation_error=e,
                ) from e

    async def _run_single_guard(
        self,
        guard: Guard,
        guard_input: GuardInput,
        index: int,
    ) -> tuple[int, bool, Exception | None]:
        """Run a single guard and return (index, passed, exception)."""
        # DEBUG: Log what's being passed to the guard
        print(f"DEBUG _run_single_guard(): index={index}, guard={guard}")
        print(f"DEBUG _run_single_guard(): guard_input type={type(guard_input).__name__}")
        print(f"DEBUG _run_single_guard(): guard_input value={guard_input}")
        if hasattr(guard_input, 'prompt'):
            print(f"DEBUG _run_single_guard(): guard_input.prompt = '{guard_input.prompt}'")
        if hasattr(guard_input, 'model_dump'):
            print(f"DEBUG _run_single_guard(): guard_input.model_dump() = {guard_input.model_dump()}")
        try:
            result = guard(guard_input)
            print(f"DEBUG _run_single_guard(): guard returned (before await): {result}")
            if asyncio.iscoroutine(result):
                result = await result
            print(f"DEBUG _run_single_guard(): guard returned (after await): {result}")
            return (index, bool(result), None)
        except Exception as e:
            print(f"DEBUG _run_single_guard(): guard raised exception: {e}")
            return (index, False, e)

    async def _run_guards_parallel(
        self,
        guard_inputs: list[GuardInput],
    ) -> list[tuple[int, bool, Exception | None]]:
        """Run all guards in parallel."""
        tasks = [
            self._run_single_guard(guard, guard_input, i)
            for i, (guard, guard_input) in enumerate(zip(self._guards, guard_inputs))
        ]
        return await asyncio.gather(*tasks)

    async def _run_guards_sequential(
        self,
        guard_inputs: list[GuardInput],
    ) -> list[tuple[int, bool, Exception | None]]:
        """Run guards sequentially, optionally stopping at first failure."""
        results = []
        for i, (guard, guard_input) in enumerate(zip(self._guards, guard_inputs)):
            result = await self._run_single_guard(guard, guard_input, i)
            results.append(result)

            _, passed, exception = result
            if (not passed or exception) and not self._run_all:
                # Stop at first failure
                break

        return results

    async def run(
        self,
        func_to_guard: Callable[
            [], Awaitable[GuardedOutput[GuardedFunctionResult, GuardInput]]
        ],
    ) -> GuardedFunctionResult | FailureResult:
        """
        Execute a function with guardrail protection.

        Must call create() first to configure guards and on_failure.

        Args:
            func_to_guard: Async function that returns GuardedOutput[T, Z].
                           The guard_inputs list must match the number of guards.

        Returns:
            T | F: The result from GuardedOutput.result, or the on_failure return value

        Raises:
            GuardValidationError: If any guard returns False and on_failure raises
            GuardExecutionError: If a guard function throws an exception
            ValueError: If create() was not called or guard_inputs length doesn't match

        Example:
            g = client.guardrails.create(
                guards=[lambda z: z["score"] > 0.8],
                on_failure=OnFailure.raise_exception("Quality check failed"),
            )
            result = await g.run(generate_email)
        """
        if not self._guards or self._on_failure is None:
            raise ValueError("Must call create() before run()")

        with get_tracer() as tracer:
            with tracer.start_as_current_span("guardrail.run") as span:
                start_time = time.perf_counter()

                # 1. Execute func_to_guard
                output: GuardedOutput[
                    GuardedFunctionResult, GuardInput
                ] = await func_to_guard()

                # 2. Validate guard_inputs (length and types)
                self._validate_inputs(output.guard_inputs)

                # 3. Run guards
                if self._parallel:
                    results = await self._run_guards_parallel(output.guard_inputs)
                else:
                    results = await self._run_guards_sequential(output.guard_inputs)

                # 4. Check for execution errors
                for index, passed, exception in results:
                    if exception is not None:
                        span.set_status(Status(StatusCode.ERROR, str(exception)))
                        span.record_exception(exception)
                        raise GuardExecutionError(
                            message="Guard execution failed",
                            original_exception=exception,
                            guard_input=output.guard_inputs[index],
                            guard_index=index,
                        ) from exception

                # 5. Check for failures
                failed_indices = [i for i, passed, _ in results if not passed]
                all_passed = len(failed_indices) == 0

                duration_ms = (time.perf_counter() - start_time) * 1000

                span.set_attribute("guardrail.passed", all_passed)
                span.set_attribute("guardrail.duration_ms", duration_ms)
                span.set_attribute("guardrail.guards_count", len(self._guards))
                if failed_indices:
                    span.set_attribute("guardrail.failed_indices", failed_indices)

                # 6. Handle failure
                if not all_passed:
                    failure_result = self._on_failure(output)
                    if asyncio.iscoroutine(failure_result):
                        failure_result = await failure_result
                    return cast(FailureResult, failure_result)

                # 7. All guards passed, return result
                return output.result

    async def validate(
        self,
        guard_inputs: list[GuardInput],
        on_failure: Optional[OnFailureHandler] = None,
    ) -> bool:
        """
        Run guards on inputs directly, without wrapping in a function.

        Must call create() first to configure guards.

        Args:
            guard_inputs: List of inputs for each guard (must match number of guards)
            on_failure: Optional handler to override the class-configured on_failure

        Returns:
            bool: True if all guards pass, False if any guard fails

        Raises:
            GuardExecutionError: If a guard function throws an exception
            ValueError: If create() was not called or guard_inputs length doesn't match

        Example:
            g = client.guardrails.create(
                guards=[lambda z: z["score"] > 0.8],
                on_failure=OnFailure.log(),
            )
            passed = await g.validate([{"score": 0.9}])  # Returns True
        """
        if not self._guards:
            raise ValueError("Must call create() before validate()")

        failure_handler = on_failure if on_failure is not None else self._on_failure

        # DEBUG: Log incoming guard_inputs
        print(f"DEBUG validate(): Received guard_inputs: {guard_inputs}")
        for i, gi in enumerate(guard_inputs):
            print(f"DEBUG validate(): guard_input[{i}] type={type(gi).__name__}, value={gi}")
            if hasattr(gi, 'prompt'):
                print(f"DEBUG validate(): guard_input[{i}].prompt = '{gi.prompt}'")
            if hasattr(gi, 'model_dump'):
                print(f"DEBUG validate(): guard_input[{i}].model_dump() = {gi.model_dump()}")

        with get_tracer() as tracer:
            with tracer.start_as_current_span("guardrail.validate") as span:
                start_time = time.perf_counter()

                # 1. Validate guard_inputs (length and types)
                self._validate_inputs(guard_inputs)

                # 2. Run guards
                if self._parallel:
                    results = await self._run_guards_parallel(guard_inputs)
                else:
                    results = await self._run_guards_sequential(guard_inputs)

                # 3. Check for execution errors
                for index, _, exception in results:
                    if exception is not None:
                        span.set_status(Status(StatusCode.ERROR, str(exception)))
                        span.record_exception(exception)
                        raise GuardExecutionError(
                            message="Guard execution failed",
                            original_exception=exception,
                            guard_input=guard_inputs[index],
                            guard_index=index,
                        ) from exception

                # 4. Check for failures
                failed_indices = [i for i, passed, _ in results if not passed]
                all_passed = len(failed_indices) == 0

                duration_ms = (time.perf_counter() - start_time) * 1000

                span.set_attribute("guardrail.passed", all_passed)
                span.set_attribute("guardrail.duration_ms", duration_ms)
                span.set_attribute("guardrail.guards_count", len(self._guards))
                if failed_indices:
                    span.set_attribute("guardrail.failed_indices", failed_indices)

                # 5. Handle failure
                if not all_passed and failure_handler is not None:
                    output = GuardedOutput(result=None, guard_inputs=guard_inputs)
                    failure_result = failure_handler(output)
                    if asyncio.iscoroutine(failure_result):
                        await failure_result

                # 6. Return validation result
                return all_passed
