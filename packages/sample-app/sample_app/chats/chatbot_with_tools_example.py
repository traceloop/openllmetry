"""
Enhanced chatbot example with function tools using OpenAI Agents SDK.

This example demonstrates:
- Creating a chatbot with function tools
- Tool execution and result handling
- Weather information retrieval
- Calculator functionality
- OpenTelemetry tracing for tools and agent interactions
"""

import asyncio
import datetime
import uuid
from typing import Dict
from dotenv import load_dotenv
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import conversation
from agents import Agent, Runner, function_tool, RunContextWrapper

load_dotenv()

# Initialize Traceloop for OpenTelemetry tracing
traceloop = Traceloop.init(
    app_name="openai-agents-chatbot-with-tools",
    disable_batch=False,
)


@function_tool
async def get_weather(cw: RunContextWrapper, city: str) -> Dict[str, str]:
    """
    Get current weather information for a city.

    Args:
        city: Name of the city to get weather for

    Returns:
        Dictionary with weather information
    """
    # Simulated weather data
    weather_data = {
        "London": {"temp": "15°C", "condition": "Cloudy", "humidity": "75%"},
        "New York": {"temp": "22°C", "condition": "Sunny", "humidity": "60%"},
        "Tokyo": {"temp": "18°C", "condition": "Rainy", "humidity": "85%"},
        "Paris": {"temp": "17°C", "condition": "Partly Cloudy", "humidity": "70%"},
        "Sydney": {"temp": "25°C", "condition": "Sunny", "humidity": "55%"},
    }

    # Default weather for unknown cities
    weather = weather_data.get(city, {
        "temp": "20°C",
        "condition": "Clear",
        "humidity": "65%"
    })

    return {
        "city": city,
        "temperature": weather["temp"],
        "condition": weather["condition"],
        "humidity": weather["humidity"],
        "timestamp": datetime.datetime.now().isoformat()
    }


@function_tool
async def calculate(cw: RunContextWrapper, expression: str) -> Dict[str, str]:
    """
    Calculate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")

    Returns:
        Dictionary with calculation result
    """
    try:
        # Safe evaluation of basic math expressions
        # Only allow specific operators
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return {
                "expression": expression,
                "result": "Error: Invalid characters in expression",
                "success": False
            }

        result = eval(expression, {"__builtins__": {}}, {})
        return {
            "expression": expression,
            "result": str(result),
            "success": True
        }
    except Exception as e:
        return {
            "expression": expression,
            "result": f"Error: {str(e)}",
            "success": False
        }


@function_tool
async def get_current_time(cw: RunContextWrapper, timezone: str = "UTC") -> Dict[str, str]:
    """
    Get the current time.

    Args:
        timezone: Timezone name (currently only returns UTC)

    Returns:
        Dictionary with current time information
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    return {
        "timezone": timezone,
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "datetime": now.isoformat()
    }


def create_assistant_agent(name: str = "Assistant", model: str = "gpt-4o-mini") -> Agent:
    """
    Create an enhanced chatbot agent with function tools.

    Args:
        name: Name of the assistant agent
        model: OpenAI model to use

    Returns:
        Configured Agent instance with tools
    """
    return Agent(
        name=name,
        instructions="""
        You are a helpful assistant with access to several tools:
        - Weather information lookup
        - Mathematical calculations
        - Current time information

        When users ask about weather, use the get_weather tool.
        When users ask for calculations, use the calculate tool.
        When users ask for the time, use the get_current_time tool.

        Be conversational and helpful. Explain the results from your tools clearly.
        """,
        model=model,
        tools=[get_weather, calculate, get_current_time],
    )


@conversation(conversation_id=str(uuid.uuid4()))
async def main():
    """Main function for interactive chatbot.

    Note: This function uses the @conversation decorator with a random UUID
    to track all spans within this conversation session.
    """
    assistant = create_assistant_agent()

    print("Starting chatbot conversation")
    print("Type 'exit', 'quit', or 'bye' to end the conversation")
    print("=" * 80)

    messages = []

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check for exit commands
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye! Chat session ended.")
            break

        # Skip empty messages
        if not user_input:
            continue

        # Add user message to conversation history
        messages.append({"role": "user", "content": user_input})

        # Process the message
        try:
            # Run the agent with streaming
            print("\nAssistant: ", end="", flush=True)
            runner = Runner().run_streamed(starting_agent=assistant, input=messages)

            response_text = []
            tool_calls_made = []

            async for event in runner.stream_events():
                if event.type == "raw_response_event":
                    # Handle text streaming
                    if hasattr(event.data, 'delta'):
                        print(event.data.delta, end="", flush=True)
                        response_text.append(event.data.delta)
                    # Track tool calls
                    elif hasattr(event.data, 'item'):
                        if hasattr(event.data.item, 'name'):
                            tool_name = event.data.item.name
                            if tool_name not in tool_calls_made:
                                tool_calls_made.append(tool_name)
                                print(f"\n[Tool Call]: {tool_name}", end=" ", flush=True)

            print()

            # Add assistant response to conversation history
            if response_text:
                messages.append({
                    "role": "assistant",
                    "content": "".join(response_text)
                })
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


if __name__ == "__main__":
    asyncio.run(main())
