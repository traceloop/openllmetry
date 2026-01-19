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
    GuardedFunctionOutput,
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

    async def generate_summary() -> GuardedFunctionOutput[str, dict]:
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
        return GuardedFunctionOutput(
            result=text,
            guard_input={"text": text, "word_count": len(text.split())},
        )

    try:
        result = await client.guardrails.run(
            func_to_guard=generate_summary,
            guard=lambda z: z["word_count"] < 100,  # Max 100 words
            on_failure=OnFailure.raise_exception("Summary too long"),
        )
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

    async def generate_travel_advice() -> GuardedFunctionOutput[str, dict]:
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
        return GuardedFunctionOutput(
            result=text,
            guard_input={"text": text},
        )

    try:
        result = await client.guardrails.run(
            func_to_guard=generate_travel_advice,
            guard=content_safety_guard,  # Pass function reference
            on_failure=OnFailure.raise_exception("Content failed safety checks"),
        )
        print(f"Travel advice: {result[:100]}...")
    except GuardValidationError as e:
        print(f"Guard failed: {e}")


# Example 3: Custom on_failure Handler
# ===================================
@workflow(name="custom_handler_example")
async def custom_handler_example():
    """Demonstrate custom on_failure handler with logging and alerting."""

    async def generate_content() -> GuardedFunctionOutput[str, dict]:
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
        return GuardedFunctionOutput(
            result=text,
            guard_input={"text": text},
        )

    def custom_alert_handler(output: GuardedFunctionOutput) -> None:
        """Custom handler that logs and could send alerts."""
        print(f"[ALERT] Guard failed for output: {str(output.result)[:50]}...")
        print(f"[ALERT] Guard input was: {list(output.guard_input.keys())}")
        # In production, you might:
        # - Send to Slack/PagerDuty
        # - Log to monitoring system
        # - Store for review
        raise GuardValidationError("Blocked after alerting team", output)

    try:
        result = await client.guardrails.run(
            func_to_guard=generate_content,
            guard=lambda z: "danger" not in z["text"].lower(),
            on_failure=custom_alert_handler,
        )
        print(f"Content: {result[:100]}...")
    except GuardValidationError:
        print("Response was blocked by custom handler")


# Example 4: Shadow Mode (Evaluate but Don't Block)
# ================================================
@workflow(name="shadow_mode_example")
async def shadow_mode_example():
    """Demonstrate shadow mode for testing guards in production."""

    async def generate_response() -> GuardedFunctionOutput[str, dict]:
        """Generate a response to test with experimental guard."""
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Tell me about visa requirements for Europe."}
            ],
        )
        text = completion.choices[0].message.content
        return GuardedFunctionOutput(
            result=text,
            guard_input={"text": text, "length": len(text)},
        )

    # Shadow mode: evaluate but don't block
    result = await client.guardrails.run(
        func_to_guard=generate_response,
        guard=lambda z: z["length"] > 50,  # Arbitrary check
        on_failure=OnFailure.noop(),  # Just observe, don't block
    )
    print(f"Response (shadow mode, always returns result): {result[:100]}...")


# Example 5: Async Guard Function
# ==============================
async def async_content_validator(guard_input: dict) -> bool:
    """
    Async guard function that could call external services.

    In a real scenario, this might:
    - Call an external moderation API
    - Query a database for blocklists
    - Perform async ML inference
    """
    text = guard_input.get("text", "")

    # Simulate async operation (e.g., external API call)
    await asyncio.sleep(0.1)

    # Simple validation
    word_count = len(text.split())
    return 10 <= word_count <= 500


@workflow(name="async_guard_example")
async def async_guard_example():
    """Demonstrate an async guard function."""

    async def generate_article() -> GuardedFunctionOutput[str, dict]:
        """Generate a travel article."""
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": "Write a short paragraph about backpacking in Vietnam.",
                }
            ],
        )
        text = completion.choices[0].message.content
        return GuardedFunctionOutput(
            result=text,
            guard_input={"text": text},
        )

    result = await client.guardrails.run(
        func_to_guard=generate_article,
        guard=async_content_validator,  # Async guard function
        on_failure=OnFailure.raise_exception("Content validation failed"),
    )
    print(f"Article: {result[:100]}...")


# Example 6: Guard with Return Value on Failure
# ============================================
@workflow(name="fallback_value_example")
async def fallback_value_example():
    """Demonstrate returning a fallback value instead of raising on failure."""

    async def generate_recommendation() -> GuardedFunctionOutput[str, dict]:
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
        return GuardedFunctionOutput(
            result=text,
            guard_input={"text": text},
        )

    # Return a safe fallback instead of raising
    result = await client.guardrails.run(
        func_to_guard=generate_recommendation,
        guard=lambda z: "risk" not in z["text"].lower(),
        on_failure=OnFailure.return_value(
            "I'd recommend visiting a local museum or taking a guided walking tour - "
            "both are safe and enjoyable options!"
        ),
    )
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
    await shadow_mode_example()

    print("\n" + "=" * 60)
    print("Example 5: Async Guard Function")
    print("=" * 60)
    await async_guard_example()

    print("\n" + "=" * 60)
    print("Example 6: Guard with Fallback Value on Failure")
    print("=" * 60)
    await fallback_value_example()


if __name__ == "__main__":
    asyncio.run(main())
