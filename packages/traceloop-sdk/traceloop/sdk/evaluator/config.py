from typing import Dict, Any, Optional, List, Callable, Awaitable
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
        timeout_in_sec: int = 60,
    ) -> Callable[[BaseModel | dict], Awaitable[bool]]:
        """
        Convert this evaluator to a guard function for use with client.guardrails.run().

        Args:
            condition: Function that receives evaluator result and returns bool.
                       True = pass, False = fail.
                       Use Condition helpers (e.g., Condition.success(), Condition.score_above(0.8))
                       or a custom lambda.
            timeout_in_sec: Maximum time to wait for evaluator execution (default: 60 seconds)

        Returns:
            Async function suitable for client.guardrails.run(guard=...)
            Accepts either a Pydantic BaseModel or a plain dict as input.

        Example with Pydantic Model:
            from traceloop.sdk import Traceloop
            from traceloop.sdk.guardrail import Condition, OnFailure, GuardedFunctionOutput
            from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop
            from traceloop.sdk.generated.evaluators import ToxicityDetectorInput

            client = Traceloop.init(api_key="...")

            async def generate_text() -> GuardedFunctionOutput[str, ToxicityDetectorInput]:
                text = "Hello world"
                return GuardedFunctionOutput(
                    result=text,
                    guard_input=ToxicityDetectorInput(text=text),
                )

            result = await client.guardrails.run(
                func_to_guard=generate_text,
                guard=EvaluatorMadeByTraceloop.toxicity_detector().as_guard(
                    condition=Condition.is_false("is_toxic")
                ),
                on_failure=OnFailure.raise_exception("Toxic content detected"),
            )

        Example with Dictionary:
            from pydantic import BaseModel

            class CustomInput(BaseModel):
                text: str
                context: str

            async def generate_text() -> GuardedFunctionOutput[str, CustomInput]:
                text = "Hello world"
                return GuardedFunctionOutput(
                    result=text,
                    guard_input=CustomInput(text=text, context="greeting"),
                )

            # Or use a plain dict:
            async def generate_text_dict() -> GuardedFunctionOutput[str, dict]:
                text = "Hello world"
                return GuardedFunctionOutput(
                    result=text,
                    guard_input={"text": text, "context": "greeting"},
                )
        """
        evaluator_slug = self.slug
        evaluator_version = self.version
        evaluator_config = self.config

        async def guard_fn(input_data) -> bool:
            # Lazy import to avoid circular dependencies
            from traceloop.sdk import Traceloop
            from traceloop.sdk.evaluator.evaluator import Evaluator

            # Convert Pydantic model to dict, or use dict directly
            if isinstance(input_data, dict):
                input_dict = input_data
            elif hasattr(input_data, "model_dump"):
                input_dict = input_data.model_dump()
            else:
                # Fallback: try to convert to dict
                input_dict = dict(input_data)

            # Get the SDK client
            client = Traceloop.get()
            evaluator = Evaluator(client._async_http)

            eval_response = await evaluator.run(
                evaluator_slug=evaluator_slug,
                input=input_dict,
                evaluator_version=evaluator_version,
                evaluator_config=evaluator_config,
                timeout_in_sec=timeout_in_sec,
            )
            print(f"NOMI - Evaluator response in the guard function: {eval_response}")

            condition_result = condition(eval_response.result.evaluator_result)
            print(f"NOMI - Condition result: {condition_result}")
            return condition_result

        # Set the function name to the evaluator slug for tracing
        guard_fn.__name__ = evaluator_slug

        return guard_fn
