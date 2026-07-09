"""
Guardrail Example: Using Traceloop Evaluators as Guards

This example demonstrates how to use Traceloop's built-in evaluators
(PII detection, toxicity, agent goal completeness) as guards.

Requires a Traceloop API key to run the evaluators.
"""

import asyncio
import os

from openai import AsyncOpenAI

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
from traceloop.sdk.guardrail import (
    Guardrails,
    pii_guard,
    toxicity_guard,
)
from traceloop.sdk.generated.evaluators.request import (
    ToxicityDetectorInput,
    PIIDetectorInput,
)

# Initialize Traceloop
Traceloop.init(app_name="guardrail-traceloop-evaluator", disable_batch=True, endpoint_is_traceloop=True)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Example 1: PII Detection Guard
# ==============================

async def generate_customer_response() -> str:
    """Generate a customer service response."""
    completion = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful travel agent. Never include personal information.",
            },
            {
                "role": "user",
                "content": "What are the best beaches in Thailand?",
            },
        ],
    )
    return completion.choices[0].message.content or ""


@workflow(name="pii_guard_example")
async def pii_guard_example():
    """Demonstrate PII detection guard using Traceloop evaluator."""

    guardrail = Guardrails(
        pii_guard(probability_threshold=0.7, timeout_in_sec=45),
        on_failure="raise",
    )
    result = await guardrail.run(
        generate_customer_response,
        input_mapper=lambda text: [PIIDetectorInput(text=text)],
    )
    print(f"Customer response: {result[:100]}...")


# Example 2: Toxicity Detection Guard
# ===================================
async def generate_content() -> str:
    """Generate travel content that should be family-friendly."""
    completion = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Write a family-friendly description of nightlife in Tokyo.",
            }
        ],
    )
    return completion.choices[0].message.content or ""


@workflow(name="toxicity_guard_example")
async def toxicity_guard_example():
    """Demonstrate toxicity detection with score-based condition."""

    guardrail = Guardrails(
        toxicity_guard(threshold=0.7),
        on_failure="raise",
    )
    result = await guardrail.run(
        generate_content,
        input_mapper=lambda text: [ToxicityDetectorInput(text=text)],
    )
    print(f"Family-friendly content: {result[:100]}...")


async def main():
    """Run all Traceloop evaluator guard examples."""
    print("=" * 60)
    print("Example 1: PII Detection Guard")
    print("=" * 60)
    try:
        await pii_guard_example()
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Example 2: Toxicity Detection Guard")
    print("=" * 60)
    try:
        await toxicity_guard_example()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
