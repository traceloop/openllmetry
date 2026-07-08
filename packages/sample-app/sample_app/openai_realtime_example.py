"""
OpenAI Realtime API example using WebSocket connection.

This example demonstrates how to use the OpenAI Realtime API with tracing instrumentation.
The Realtime API allows low-latency, multi-modal conversations over WebSocket.

Requires:
- OpenAI Python SDK >= 1.0.0
- Valid OPENAI_API_KEY environment variable

Usage:
    OPENAI_API_KEY=your-key python openai_realtime_example.py
"""

import asyncio
import os

from openai import AsyncOpenAI

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow


# Initialize tracing
Traceloop.init(app_name="openai_realtime_demo")

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@workflow(name="realtime_text_conversation")
async def realtime_text_conversation():
    """
    Demonstrates a simple text-based conversation using the Realtime API.

    The Realtime API supports:
    - Text and audio input/output
    - Function/tool calling
    - Session configuration updates
    - Low-latency streaming responses
    """
    async with client.beta.realtime.connect(
        model="gpt-4o-realtime-preview-2024-12-17",
    ) as connection:
        # Update session configuration
        await connection.session.update(
            session={
                "modalities": ["text"],
                "instructions": "You are a helpful assistant. Be concise.",
                "temperature": 0.7,
            }
        )

        # Wait for session update confirmation
        async for event in connection:
            if event.type == "session.updated":
                print(f"Session updated: modalities={event.session.modalities}")
                break

        # Send a user message
        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "What is the capital of France? Answer in one word.",
                    }
                ],
            }
        )

        # Request a response
        await connection.response.create()

        # Collect the response
        response_text = ""
        async for event in connection:
            if event.type == "response.text.delta":
                response_text += event.delta
                print(event.delta, end="", flush=True)
            elif event.type == "response.text.done":
                print()  # newline after response
            elif event.type == "response.done":
                print("\n[Response completed]")
                if hasattr(event.response, "usage"):
                    usage = event.response.usage
                    print(
                        f"Usage: input={usage.input_tokens}, output={usage.output_tokens}"
                    )
                break
            elif event.type == "error":
                print(f"Error: {event.error}")
                break

        return response_text


@workflow(name="realtime_multi_turn_conversation")
async def realtime_multi_turn_conversation():
    """
    Demonstrates a multi-turn conversation using the Realtime API.
    """
    async with client.beta.realtime.connect(
        model="gpt-4o-realtime-preview-2024-12-17",
    ) as connection:
        # Configure session for text-only mode
        await connection.session.update(
            session={
                "modalities": ["text"],
                "instructions": "You are a math tutor. Be helpful and explain your reasoning.",
            }
        )

        # Wait for session update
        async for event in connection:
            if event.type == "session.updated":
                break

        # First turn: Ask a question
        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What is 2 + 2?"}],
            }
        )
        await connection.response.create()

        print("Turn 1:")
        async for event in connection:
            if event.type == "response.text.delta":
                print(event.delta, end="", flush=True)
            elif event.type == "response.done":
                print("\n")
                break

        # Second turn: Follow-up question
        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "And what is that times 3?"}
                ],
            }
        )
        await connection.response.create()

        print("Turn 2:")
        async for event in connection:
            if event.type == "response.text.delta":
                print(event.delta, end="", flush=True)
            elif event.type == "response.done":
                print("\n")
                break


@workflow(name="realtime_with_tools")
async def realtime_with_tools():
    """
    Demonstrates function calling with the Realtime API.
    """
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., 'London'",
                    }
                },
                "required": ["location"],
            },
        }
    ]

    async with client.beta.realtime.connect(
        model="gpt-4o-realtime-preview-2024-12-17",
    ) as connection:
        # Configure session with tools
        await connection.session.update(
            session={
                "modalities": ["text"],
                "tools": tools,
                "tool_choice": "auto",
            }
        )

        # Wait for session update
        async for event in connection:
            if event.type == "session.updated":
                break

        # Ask about weather (should trigger tool call)
        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What's the weather in Paris?"}
                ],
            }
        )
        await connection.response.create()

        # Handle response and potential tool calls
        async for event in connection:
            if event.type == "response.function_call_arguments.done":
                print(f"Tool call: {event.name}({event.arguments})")
                # In a real app, you would execute the function and send the result back
            elif event.type == "response.text.delta":
                print(event.delta, end="", flush=True)
            elif event.type == "response.done":
                print("\n[Response completed]")
                break


async def main():
    print("=" * 60)
    print("OpenAI Realtime API Example")
    print("=" * 60)

    print("\n1. Simple text conversation:")
    print("-" * 40)
    await realtime_text_conversation()

    print("\n2. Multi-turn conversation:")
    print("-" * 40)
    await realtime_multi_turn_conversation()

    print("\n3. Conversation with tools:")
    print("-" * 40)
    await realtime_with_tools()


if __name__ == "__main__":
    asyncio.run(main())
