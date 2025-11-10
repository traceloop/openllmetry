#!/usr/bin/env python3
"""
Test script to verify that streaming responses.create() now generates traces.
This reproduces the customer issue reported in the Slack conversation.
"""
import asyncio
import os
from openai import AsyncOpenAI
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from traceloop.sdk import Traceloop

# Initialize Traceloop with console exporter for debugging
Traceloop.init(
    app_name="test-responses-streaming",
    exporter=ConsoleSpanExporter(),
    disable_batch=True,
)


async def test_responses_streaming():
    """Test the responses.create() streaming API"""
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"),
        timeout=60.0,
    )

    request_params = {
        "model": "gpt-4.1-nano",
        "input": "Please tell me a short story about a unicorn.",
        "temperature": 0.3,
        "stream": True,
    }

    print("Starting streaming request...")
    stream = await client.responses.create(**request_params)

    print("Consuming stream...")
    full_text = ""
    async for item in stream:
        if hasattr(item, "type") and item.type == "response.output_text.delta":
            if hasattr(item, "delta") and item.delta:
                print(item.delta, end="", flush=True)
                full_text += item.delta

    print("\n\nStream complete!")
    print(f"Total text length: {len(full_text)}")
    print("\nSpan should have been exported above with trace details.")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing OpenAI Responses API Streaming with Traceloop")
    print("=" * 80)
    asyncio.run(test_responses_streaming())
