"""
Guardrail Decorator Example

Demonstrates using @guardrail decorator to protect functions.
All configuration is inline - no need to create guardrail instance first.

Requires a Traceloop API key for the evaluators.
"""

import asyncio
import os

from openai import AsyncOpenAI

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
from traceloop.sdk.decorators import aguardrail as guardrail
from traceloop.sdk.guardrail import OnFailure, Guards

# Initialize Traceloop (required for @guardrail decorator)
Traceloop.init(app_name="guardrail-decorator-example", disable_batch=True)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_KEY"))


@guardrail(
    name="toxicity-response-guard",
    guards=[Guards.toxicity_detector()],
    on_failure=OnFailure.return_value("Sorry, I cannot provide that response."),
)
async def generate_response(user_prompt: str) -> str:
    """Generate LLM response - automatically guarded by decorator."""
    completion = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


@workflow(name="chat_with_decorator")
async def chat(user_prompt: str) -> str:
    """Chat function - guardrail runs automatically when generate_response is called."""
    return await generate_response(user_prompt)


async def main():
    """Run the decorator example."""

    print("=" * 60)
    print("Guardrail Decorator Example")
    print("=" * 60)

    print("\n--- Test: Safe prompt ---")
    try:
        response = await chat("What are some fun activities in Paris?")
        print(f"Response: {response[:200]}...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
