import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

from dotenv import load_dotenv

load_dotenv()

Traceloop.init(app_name="agno_example")


def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny and 72°F"


@workflow(name="agno_agent_example")
def run_agent():
    agent = Agent(
        name="WeatherAssistant",
        model=OpenAIChat(
            id="gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        tools=[get_weather],
        description="An agent that helps with weather information",
        instructions=[
            "Be helpful and concise",
            "Always use the weather tool when asked",
        ],
    )

    print("Running agent with tool call...")
    result = agent.run("What's the weather like in San Francisco?")
    print(f"\nAgent response: {result.content}")

    return result


if __name__ == "__main__":
    run_agent()
