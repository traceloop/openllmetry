from typing import Dict, Any, Optional, List, Callable, Awaitable
from uuid import uuid4
from pydantic import BaseModel


class EvaluatorDetails(BaseModel):
    """
    Details for configuring an evaluator.

    Args:
        slug: The evaluator slug/identifier
        version: Optional version of the evaluator
        config: Optional configuration dictionary for the evaluator
        required_input_fields: Optional list of required fields to the evaluator
            input. These fields must be present in the task output.

    Example:
        >>> EvaluatorDetails(slug="pii-detector", config={"probability_threshold": 0.8}, required_input_fields=["text"])
        >>> EvaluatorDetails(slug="my-custom-evaluator", version="v2")
    """

    slug: str
    version: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    required_input_fields: Optional[List[str]] = None

    def as_guard(
        self,
        condition: Callable[[Any], bool],
    ) -> Callable[[Dict[str, Any]], Awaitable[bool]]:
        """
        Convert this evaluator to a guard function for use with client.guardrails.run().

        Args:
            condition: Function that receives evaluator result and returns bool.
                       True = pass, False = fail.
                       Use Condition helpers (e.g., Condition.success(), Condition.score_above(0.8))
                       or a custom lambda.

        Returns:
            Async function suitable for client.guardrails.run(guard=...)

        Example:
            from traceloop.sdk import Traceloop
            from traceloop.sdk.guardrail import Condition, OnFailure
            from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop

            client = Traceloop.init(api_key="...")

            result = await client.guardrails.run(
                func_to_guard=my_agent,
                guard=EvaluatorMadeByTraceloop.pii_detector().as_guard(
                    condition=Condition.is_false("has_pii")
                ),
                on_failure=OnFailure.raise_exception("PII detected"),
            )
        """
        evaluator_slug = self.slug
        evaluator_version = self.version
        evaluator_config = self.config

        async def guard_fn(input_data: Dict[str, Any]) -> bool:
            # Lazy import to avoid circular dependencies
            from traceloop.sdk import Traceloop
            from traceloop.sdk.evaluator.evaluator import Evaluator

            # Get the SDK client
            client = Traceloop.get()
            evaluator = Evaluator(client._async_http)

            # Generate internal IDs for guardrail execution (no experiment context)
            internal_task_id = f"guard-{uuid4().hex[:8]}"
            internal_experiment_id = f"guard-exp-{uuid4().hex[:8]}"
            internal_run_id = f"guard-run-{uuid4().hex[:8]}"

            try:
                eval_response = await evaluator.run_experiment_evaluator(
                    evaluator_slug=evaluator_slug,
                    task_id=internal_task_id,
                    experiment_id=internal_experiment_id,
                    experiment_run_id=internal_run_id,
                    input=input_data,
                    evaluator_version=evaluator_version,
                    evaluator_config=evaluator_config,
                    timeout_in_sec=60,
                )
                # Apply condition to the result
                return condition(eval_response)
            except Exception:
                # Evaluator execution failed - treat as guard failure
                return False

        return guard_fn
