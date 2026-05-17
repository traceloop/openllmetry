import asyncio
import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from traceloop.sdk import Traceloop

Traceloop.init(app_name="agno_streaming_example")


def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny and 72 degrees."


async def test_streaming():
    agent = Agent(
        name="WeatherAgent",
        model=OpenAIChat(
            id="gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        tools=[get_weather],
        description="An agent that provides weather information",
    )

    print("Testing async streaming with Traceloop instrumentation...")
    print("=" * 60)

    async for event in agent.arun(
        "What's the weather like in San Francisco?",
        stream=True
    ):
        if hasattr(event, "event"):
            print(f"Event: {event.event}")
        if hasattr(event, "content") and event.content:
            print(f"Content: {event.content}")
        print("-" * 40)

    print("=" * 60)
    print("Streaming completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_streaming())
