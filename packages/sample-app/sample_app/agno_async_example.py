import asyncio
import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

Traceloop.init(app_name="agno_async_example")


def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b


def calculate_product(a: int, b: int) -> int:
    """Calculate the product of two numbers."""
    return a * b


@workflow(name="agno_async_agent_example")
async def run_async_agent():
    agent = Agent(
        name="MathAssistant",
        model=OpenAIChat(
            id="gpt-4o-mini",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        tools=[calculate_sum, calculate_product],
        description="An agent that helps with mathematical calculations",
        instructions=["Use the provided tools to perform calculations"],
    )

    print("Running async agent with tool calls...")
    result = await agent.arun(
        "What is the sum of 15 and 27? Also, what is the product of 8 and 9?"
    )
    print(f"\nAgent response: {result.content}")

    return result


if __name__ == "__main__":
    asyncio.run(run_async_agent())
