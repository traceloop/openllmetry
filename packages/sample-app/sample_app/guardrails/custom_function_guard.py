"""
Guardrail Example: Using Custom Functions as Guards

This example demonstrates how to use custom functions (lambdas, regular functions,
async functions) as guards. No Traceloop API key required for these examples.

Examples shown:
1. Simple lambda guard for validation
2. Custom function with complex logic
3. Custom on_failure handler with alerting
4. Shadow mode (evaluate but don't block)
5. Async guard function
"""

import asyncio
import os

from openai import AsyncOpenAI

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
from traceloop.sdk.guardrail import (
    GuardedOutput,
    OnFailure,
    GuardValidationError,
)

# Initialize Traceloop - returns client with guardrails access
client = Traceloop.init(app_name="guardrail-custom-function", disable_batch=True)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Example 1: Simple Lambda Guard
# ==============================
@workflow(name="simple_lambda_guard")
async def simple_lambda_guard_example():
    """Demonstrate a simple lambda-based guard for length validation."""

    async def generate_summary() -> GuardedOutput[str, dict]:
        """Generate a travel summary with length constraints."""
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Write a brief 2-sentence summary of Paris as a travel destination.",
                }
            ],
        )
        text = completion.choices[0].message.content
        return GuardedOutput(
            result=text,
            guard_input={"text": text, "word_count": len(text.split())},
        )

    try:
        guardrail = client.guardrails.create(
            guard=lambda z: z["word_count"] < 100,
            on_failure=OnFailure.raise_exception("Summary too long"),
        )
        result = await guardrail.run(generate_summary)
        print(f"Summary (passed guard): {result}")
    except GuardValidationError as e:
        print(f"Guard failed: {e}")


# Example 2: Custom Function Guard with Complex Logic
# ==================================================
def content_safety_guard(guard_input: dict) -> bool:
    """
    Custom guard function with complex validation logic.

    Checks multiple conditions:
    - No forbidden words
    - Minimum content length
    - No excessive capitalization
    """
    text = guard_input.get("text", "")

    # Check for forbidden words
    forbidden_words = ["dangerous", "illegal", "unsafe"]
    if any(word in text.lower() for word in forbidden_words):
        return False

    # Check minimum length
    if len(text.split()) < 10:
        return False

    # Check for excessive capitalization (shouting)
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if uppercase_ratio > 0.3:
        return False

    return True


@workflow(name="custom_function_guard")
async def custom_function_guard_example():
    """Demonstrate a custom function guard with complex logic."""

    async def generate_travel_advice() -> GuardedOutput[str, dict]:
        """Generate travel advice."""
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "What should I know about traveling to a remote area?",
                }
            ],
        )
        text = completion.choices[0].message.content
        return GuardedOutput(
            result=text,
            guard_input={"text": text},
        )

    try:
        guardrail = client.guardrails.create(
            guard=content_safety_guard,  # Pass function reference
            on_failure=OnFailure.raise_exception("Content failed safety checks"),
        )
        result = await guardrail.run(generate_travel_advice)
        print(f"Travel advice: {result[:100]}...")
    except GuardValidationError as e:
        print(f"Guard failed: {e}")


# Example 3: Custom on_failure Handler
# ===================================
@workflow(name="custom_handler_example")
async def custom_handler_example():
    """Demonstrate custom on_failure handler with logging and alerting."""

    async def generate_content() -> GuardedOutput[str, dict]:
        """Generate content that might need review."""
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Give me some extreme sports recommendations.",
                }
            ],
        )
        text = completion.choices[0].message.content
        return GuardedOutput(
            result=text,
            guard_input={"text": text},
        )

    def custom_alert_handler(output: GuardedOutput) -> None:
        """Custom handler that logs and could send alerts."""
        print(f"[ALERT] Guard failed for output: {str(output.result)[:50]}...")
        print(f"[ALERT] Guard input was: {list(output.guard_input.keys())}")
        raise GuardValidationError("Blocked after alerting team", output)

    try:
        guardrail = client.guardrails.create(
            guard=lambda z: "danger" not in z["text"].lower(),
            on_failure=custom_alert_handler,
        )
        result = await guardrail.run(generate_content)
        print(f"Content: {result[:100]}...")
    except GuardValidationError:
        print("Response was blocked by custom handler")



# Example 4: Guard with Return Value on Failure
# ============================================
@workflow(name="fallback_value_example")
async def fallback_value_example():
    """Demonstrate returning a fallback value instead of raising on failure."""

    async def generate_recommendation() -> GuardedOutput[str, dict]:
        """Generate a recommendation that might fail the guard."""
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Recommend something exciting but potentially risky to do.",
                }
            ],
        )
        text = completion.choices[0].message.content
        return GuardedOutput(
            result=text,
            guard_input={"text": text},
        )

    # Return a safe fallback instead of raising
    guardrail = client.guardrails.create(
        guard=lambda z: "risk" not in z["text"].lower(),
        on_failure=OnFailure.return_value(
            "I'd recommend visiting a local museum or taking a guided walking tour - "
            "both are safe and enjoyable options!"
        ),
    )
    result = await guardrail.run(generate_recommendation)
    print(f"Recommendation: {result}")


async def main():
    """Run all custom function guard examples."""
    print("=" * 60)
    print("Example 1: Simple Lambda Guard")
    print("=" * 60)
    await simple_lambda_guard_example()

    print("\n" + "=" * 60)
    print("Example 2: Custom Function Guard with Complex Logic")
    print("=" * 60)
    await custom_function_guard_example()

    print("\n" + "=" * 60)
    print("Example 3: Custom on_failure Handler")
    print("=" * 60)
    await custom_handler_example()

    print("\n" + "=" * 60)
    print("Example 4: Shadow Mode (Evaluate but Don't Block)")
    print("=" * 60)
    await fallback_value_example()


if __name__ == "__main__":
    asyncio.run(main())
