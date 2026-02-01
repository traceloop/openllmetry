"""
Guardrail Example: Multiple Guards in a Single Guardrail

This example demonstrates how to use multiple guards together:
1. Multiple lambda guards running in parallel
2. Mixed guard types (Evaluator + custom function)
3. Run all guards to collect all failures
4. Sequential execution for dependent guards

Key concepts:
- input_mapper returns a list matching the number of guards
- Each guard receives its corresponding input (index-matched)
- parallel=True (default) runs guards concurrently
- run_all=True continues after failures to collect all errors
"""

import asyncio
import os

from openai import AsyncOpenAI

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
from traceloop.sdk.guardrail import (
    OnFailure,
    GuardValidationError,
    GuardedResult,
    Guards,
)
from traceloop.sdk.generated.evaluators.request import PIIDetectorInput

# Initialize Traceloop
client = Traceloop.init(app_name="guardrail-multiple-guards", disable_batch=True)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Example 1: Multiple Lambda Guards (Parallel)
# ============================================
async def generate_content() -> str:
    """Generate content."""
    completion = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Write a brief travel tip about visiting Japan.",
            }
        ],
    )
    return completion.choices[0].message.content


@workflow(name="multiple_lambda_guards")
async def multiple_lambda_guards_example():
    """
    Demonstrate multiple lambda guards running in parallel.

    Each guard gets its own input from the input_mapper return list.
    Guards run concurrently by default (parallel=True).
    """

    def create_guard_inputs(text: str) -> list[dict]:
        """Create inputs for each guard."""
        word_count = len(text.split())
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / max(len(text), 1)

        # Each guard gets its own input - order must match guards list
        return [
            {"word_count": word_count},  # For length guard
            {"text": text},               # For forbidden words guard
            {"caps_ratio": caps_ratio},   # For capitalization guard
        ]

    guardrail = client.guardrails.create(
        guards=[
            lambda z: z["word_count"] < 200,           # Guard 0: Length check
            lambda z: "danger" not in z["text"].lower(),  # Guard 1: Forbidden words
            lambda z: z["caps_ratio"] < 0.3,           # Guard 2: No excessive caps
        ],
        on_failure=OnFailure.raise_exception("Content failed validation checks"),
        parallel=True,
    )

    try:
        result = await guardrail.run(generate_content, input_mapper=create_guard_inputs)
        print(f"Content passed all 3 guards: {result[:100]}...")
    except GuardValidationError as e:
        print(f"Guard failed: {e}")


# Example 2: Mixed Guard Types (Evaluator + Custom Function)
# ==========================================================
def business_rules_guard(guard_input: dict) -> bool:
    """
    Custom guard for business-specific validation.

    Checks that content meets company guidelines:
    - Mentions safety disclaimer if discussing activities
    - Doesn't make promises or guarantees
    """
    text = guard_input.get("text", "").lower()

    # Check for prohibited promises
    prohibited_phrases = ["guaranteed", "100%", "promise", "definitely will"]
    if any(phrase in text for phrase in prohibited_phrases):
        return False

    return True


async def generate_customer_response() -> str:
    """Generate a customer service response."""
    completion = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful travel agent. Be helpful but don't make guarantees.",
            },
            {
                "role": "user",
                "content": "What's the weather like in Hawaii in December?",
            },
        ],
    )
    return completion.choices[0].message.content


@workflow(name="mixed_guard_types")
async def mixed_guard_types_example():
    """
    Demonstrate mixing Traceloop evaluators with custom functions.

    Guard 0: Traceloop PII detector (evaluator-based)
    Guard 1: Custom business rules (function-based)
    """

    def create_mixed_inputs(text: str) -> list:
        """Create inputs for mixed guard types."""
        return [
            PIIDetectorInput(text=text),  # Guard 0: Evaluator input (Pydantic)
            {"text": text},                # Guard 1: Custom function input (dict)
        ]

    guardrail = client.guardrails.create(
        guards=[
            Guards.pii_detector(probability_threshold=0.7),
            business_rules_guard,
        ],
        on_failure=OnFailure.raise_exception("Response failed safety or business rules"),
    )

    try:
        result = await guardrail.run(generate_customer_response, input_mapper=create_mixed_inputs)
        print(f"Response passed PII + business rules: {result[:100]}...")
    except GuardValidationError as e:
        print(f"Guard failed: {e}")


# Example 3: Run All Guards (Collect All Failures)
# ================================================
@workflow(name="run_all_guards")
async def run_all_guards_example():
    """
    Demonstrate run_all=True to collect all failures.

    By default, guardrails stop at the first failure.
    With run_all=True, all guards run and you can see all failures.
    Useful for providing comprehensive feedback to users.
    """

    async def generate_problematic_content() -> str:
        """Generate content that might fail multiple guards."""
        # Simulate content that fails multiple checks
        return "VISIT DANGEROUS PLACES! It's guaranteed to be exciting!"

    def create_inputs(text: str) -> list[dict]:
        return [
            {"word_count": len(text.split())},
            {"text": text},
            {"has_caps_issues": text.isupper()},
        ]

    def custom_handler(output: GuardedResult) -> str:
        """Custom handler that could inspect which guards failed."""
        print(f"[Handler] Content failed validation: {output.result[:50]}...")
        return "Content did not meet our quality standards. Please try again."

    guardrail = client.guardrails.create(
        guards=[
            lambda z: z["word_count"] > 5,                    # Min length
            lambda z: "dangerous" not in z["text"].lower(),   # No dangerous
            lambda z: not z["has_caps_issues"],               # No all caps
        ],
        on_failure=custom_handler,
        run_all=True,  # Run ALL guards even after first failure
        parallel=True,
    )

    result = await guardrail.run(generate_problematic_content, input_mapper=create_inputs)
    print(f"Fallback result: {result}")


# Example 4: Sequential Guards
# ============================
@workflow(name="sequential_guards")
async def sequential_guards_example():
    """
    Demonstrate sequential guard execution with parallel=False.

    Guards run one after another. If a guard fails and run_all=False,
    subsequent guards are skipped.

    Use case: When you want to fail fast or when guards have dependencies.
    """

    async def generate_content() -> str:
        """Generate content for sequential validation."""
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Give me a one-sentence tip for visiting Tokyo.",
                }
            ],
        )
        return completion.choices[0].message.content

    def create_sequential_inputs(text: str) -> list[dict]:
        return [
            {"text": text, "step": "pre-check"},   # Step 1: Basic validation
            {"text": text, "step": "main"},        # Step 2: Main check
            {"text": text, "step": "post-check"},  # Step 3: Final check
        ]

    def pre_check(z: dict) -> bool:
        """First guard: Basic format validation."""
        print(f"  Running {z['step']}...")
        return len(z["text"]) > 0

    def main_check(z: dict) -> bool:
        """Second guard: Content validation."""
        print(f"  Running {z['step']}...")
        return "tokyo" in z["text"].lower() or "japan" in z["text"].lower()

    def post_check(z: dict) -> bool:
        """Third guard: Final quality check."""
        print(f"  Running {z['step']}...")
        return len(z["text"].split()) >= 3

    guardrail = client.guardrails.create(
        guards=[pre_check, main_check, post_check],
        on_failure=OnFailure.raise_exception("Sequential validation failed"),
        parallel=False,  # Run guards one at a time, in order
        run_all=False,   # Stop at first failure (default)
    )

    try:
        result = await guardrail.run(generate_content, input_mapper=create_sequential_inputs)
        print(f"All sequential guards passed: {result[:100]}...")
    except GuardValidationError as e:
        print(f"Guard failed: {e}")


async def main():
    """Run all multiple guards examples."""
    # print("=" * 70)
    # print("Example 1: Multiple Lambda Guards (Parallel)")
    # print("=" * 70)
    # print("3 guards checking: length, forbidden words, capitalization\n")
    # await multiple_lambda_guards_example()

    print("\n" + "=" * 70)
    print("Example 2: Mixed Guard Types (Evaluator + Custom Function)")
    print("=" * 70)
    print("Combining Traceloop PII detector with custom business rules\n")
    try:
        await mixed_guard_types_example()
    except Exception as e:
        print(f"Skipped (requires API key): {e}")

    # print("\n" + "=" * 70)
    # print("Example 3: Run All Guards (Collect All Failures)")
    # print("=" * 70)
    # print("Using run_all=True to run all guards even after failures\n")
    # await run_all_guards_example()

    # print("\n" + "=" * 70)
    # print("Example 4: Sequential Guards")
    # print("=" * 70)
    # print("Using parallel=False for ordered execution\n")
    # await sequential_guards_example()


if __name__ == "__main__":
    asyncio.run(main())
