"""
Guardrail Example: Using Custom Evaluators as Guards

This example demonstrates how to use custom evaluators (defined in Traceloop)
as guards. Custom evaluators are user-defined evaluation logic that runs
on Traceloop's infrastructure.

Requires a Traceloop API key and custom evaluators configured in your account.
"""

import asyncio
import os

from openai import AsyncOpenAI

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
from traceloop.sdk.guardrail import (
    GuardedFunctionOutput,
    Condition,
    OnFailure,
)
from traceloop.sdk.evaluator import EvaluatorDetails

# Initialize Traceloop - returns client with guardrails access
client = Traceloop.init(app_name="guardrail-custom-evaluator", disable_batch=True)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Example 1: Custom Evaluator with EvaluatorDetails
# =================================================
@workflow(name="custom_evaluator_example")
async def custom_evaluator_example():
    """
    Demonstrate using a custom evaluator defined in Traceloop.

    To use this example, first create a custom evaluator in Traceloop
    with slug "my-custom-quality-check" that returns a 'quality_score' field.
    """

    async def generate_travel_recommendation() -> GuardedFunctionOutput[str, dict]:
        """Generate a travel recommendation."""
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a travel expert. Provide detailed recommendations.",
                },
                {
                    "role": "user",
                    "content": "Recommend a 3-day itinerary for visiting Rome.",
                },
            ],
        )
        text = completion.choices[0].message.content
        return GuardedFunctionOutput(
            result=text,
            guard_input={"text": text, "destination": "Rome"},
        )

    # Define custom evaluator by slug
    custom_evaluator = EvaluatorDetails(
        slug="my-custom-quality-check",  # Your custom evaluator slug
        version="v1",  # Optional: specify version
        config={"min_words": 100},  # Optional: evaluator config
    )

    result = await client.guardrails.run(
        func_to_guard=generate_travel_recommendation,
        guard=custom_evaluator.as_guard(
            condition=Condition.score_above(0.7, field="quality_score")
        ),
        on_failure=OnFailure.log(message="Quality check failed"),
    )
    print(f"Travel recommendation: {result[:100]}...")


# Example 2: Direct Evaluator Execution
# ====================================
@workflow(name="direct_evaluator_example")
async def direct_evaluator_example():
    """
    Demonstrate running an evaluator directly without the full guardrail flow.

    Useful for building custom guard logic or when you need more control.
    """

    async def generate_and_evaluate() -> GuardedFunctionOutput[str, dict]:
        """Generate content and build custom guard input with evaluator result."""
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Write a brief description of hiking in the Swiss Alps.",
                }
            ],
        )
        text = completion.choices[0].message.content

        # Run evaluator directly to get result
        eval_result = await client.guardrails.run_evaluator(
            input_data={"text": text},
            evaluator_slug="my-content-evaluator",  # Your custom evaluator
            timeout_in_sec=30,
        )

        return GuardedFunctionOutput(
            result=text,
            guard_input={
                "text": text,
                "eval_result": eval_result,  # Include evaluator result
            },
        )

    # Use the pre-computed evaluator result in the guard
    result = await client.guardrails.run(
        func_to_guard=generate_and_evaluate,
        guard=lambda z: z["eval_result"].get("score", 0) > 0.5,
        on_failure=OnFailure.raise_exception("Content evaluation failed"),
    )
    print(f"Evaluated content: {result[:100]}...")


# Example 3: Custom Evaluator with Fallback
# ========================================
@workflow(name="evaluator_with_fallback_example")
async def evaluator_with_fallback_example():
    """
    Demonstrate using a custom evaluator with a fallback response on failure.
    """

    async def generate_response() -> GuardedFunctionOutput[str, dict]:
        """Generate a response that will be evaluated."""
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "What are some tips for budget travel in Southeast Asia?",
                }
            ],
        )
        text = completion.choices[0].message.content
        return GuardedFunctionOutput(
            result=text,
            guard_input={"text": text},
        )

    custom_evaluator = EvaluatorDetails(
        slug="budget-travel-validator",
        config={"check_practical_tips": True},
    )

    # Use return_value to provide a fallback instead of raising
    result = await client.guardrails.run(
        func_to_guard=generate_response,
        guard=custom_evaluator.as_guard(condition=Condition.success()),
        on_failure=OnFailure.return_value(
            "I apologize, but I couldn't generate quality travel tips. "
            "Please try asking again with more specific questions."
        ),
    )
    print(f"Response: {result[:100]}...")


async def main():
    """Run all custom evaluator guard examples."""
    print("=" * 60)
    print("Example 1: Custom Evaluator with EvaluatorDetails")
    print("=" * 60)
    print("Note: Requires custom evaluator 'my-custom-quality-check' in Traceloop")
    try:
        await custom_evaluator_example()
    except Exception as e:
        print(f"Skipped: {e}")

    print("\n" + "=" * 60)
    print("Example 2: Direct Evaluator Execution")
    print("=" * 60)
    print("Note: Requires custom evaluator 'my-content-evaluator' in Traceloop")
    try:
        await direct_evaluator_example()
    except Exception as e:
        print(f"Skipped: {e}")

    print("\n" + "=" * 60)
    print("Example 3: Custom Evaluator with Fallback")
    print("=" * 60)
    print("Note: Requires custom evaluator 'budget-travel-validator' in Traceloop")
    try:
        await evaluator_with_fallback_example()
    except Exception as e:
        print(f"Skipped: {e}")


if __name__ == "__main__":
    asyncio.run(main())
