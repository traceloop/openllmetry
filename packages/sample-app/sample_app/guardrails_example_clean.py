"""
Example usage of the guardrails decorator.

This example shows how to use the guardrails decorator to evaluate content
and make runtime decisions based on the calculated score.
"""

import asyncio

from traceloop.sdk import Traceloop
from traceloop.sdk.guardrails import InputExtractor

Traceloop.init(app_name="guardrails-example", api_key='tl_d0f5d2b32a054d8797fc2e15e102d91d')


async def main():
    """Main function to demonstrate guardrails usage."""
    # Initialize Traceloop (you would need to set TRACELOOP_API_KEY)
    
    print("Guardrails Example Starting")

    try:
        traceloop_client = Traceloop.get()
        print("Traceloop client: ", traceloop_client)
        result = await traceloop_client.guardrails.execute_evaluator("What I Hate", {"love_only": InputExtractor(source="bannans"), "love_sentence": InputExtractor(source="I love bannans because they are Yellow")})
        print("Result: ", result)
        print("Traceloop initialized successfully")
    except Exception as e:
        print(f"Traceloop initialization failed: {e}")
        print("Continuing with mock data...")


if __name__ == "__main__":
    asyncio.run(main()) 