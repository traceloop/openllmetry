"""
OpenAI Agents SDK Realtime API Example

This example demonstrates how to use the OpenAI Agents SDK's realtime
capabilities with OpenTelemetry instrumentation.

The realtime API enables:
- Real-time conversations with AI (text or voice)
- Tool calling during realtime sessions
- Agent handoffs in realtime context
- OpenTelemetry tracing for all realtime events

Usage:
    python openai_agents_realtime_example.py [text|voice|handoff]

    text    - Text-only realtime demo (default, no audio dependencies)
    voice   - Voice/audio realtime demo (requires [voice] extras)
    handoff - Agent handoff demo with multiple agents

Requirements:
- pip install openai-agents (realtime module is included)
- For actual audio processing: pip install openai-agents[voice]
- OPENAI_API_KEY environment variable
"""

import asyncio
import os
from typing import Optional
from dotenv import load_dotenv

from traceloop.sdk import Traceloop

# Load environment variables
load_dotenv()

# Initialize Traceloop for OpenTelemetry instrumentation
Traceloop.init(
    app_name="openai-agents-realtime-demo",
    disable_batch=False,
)


async def run_realtime_demo():
    """
    Run a demonstration of the OpenAI Agents SDK realtime capabilities.

    This demo shows:
    1. Creating a RealtimeAgent with tools
    2. Running a realtime session
    3. Handling various event types
    4. Tool execution during voice sessions
    """
    try:
        from agents import function_tool
        from agents.realtime import RealtimeAgent, RealtimeRunner
    except ImportError:
        print("Error: openai-agents is required for this example")
        print("Install with: pip install openai-agents")
        return

    print("=" * 60)
    print("OpenAI Agents SDK - Realtime Voice Demo")
    print("=" * 60)

    # Define tools for the realtime agent
    @function_tool
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        # Simulated weather data
        weather_data = {
            "new york": "Partly cloudy, 65°F",
            "london": "Rainy, 52°F",
            "tokyo": "Sunny, 78°F",
            "paris": "Cloudy, 58°F",
        }
        city_lower = city.lower()
        if city_lower in weather_data:
            return f"The weather in {city} is {weather_data[city_lower]}"
        return f"Weather data not available for {city}"

    @function_tool
    def get_time(timezone: Optional[str] = None) -> str:
        """Get current time, optionally for a specific timezone."""
        from datetime import datetime
        now = datetime.now()
        if timezone:
            return f"The current time in {timezone} is approximately {now.strftime('%I:%M %p')}"
        return f"The current time is {now.strftime('%I:%M %p')}"

    @function_tool
    def set_reminder(message: str, minutes: int) -> str:
        """Set a reminder for a specified number of minutes from now."""
        return f"Reminder set: '{message}' in {minutes} minutes"

    # Create the realtime agent with tools
    assistant = RealtimeAgent(
        name="Voice Assistant",
        instructions="""You are a helpful voice assistant. You can:
- Tell users the weather in various cities
- Provide the current time
- Set reminders for them

Keep your responses brief and conversational since this is a voice interface.
When users ask about the weather, use the get_weather tool.
When they ask about the time, use the get_time tool.
When they want to set a reminder, use the set_reminder tool.""",
        tools=[get_weather, get_time, set_reminder],
    )

    print("\nCreated Voice Assistant with tools:")
    print("  - get_weather: Get current weather for a city")
    print("  - get_time: Get current time")
    print("  - set_reminder: Set a reminder")

    # Configure the realtime runner
    runner = RealtimeRunner(
        starting_agent=assistant,
        config={
            "model_settings": {
                "model_name": "gpt-4o-realtime-preview",
                "voice": "alloy",
                "modalities": ["text", "audio"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
            }
        },
    )

    print("\nRunner configured with:")
    print("  - Model: gpt-4o-realtime-preview")
    print("  - Voice: alloy")
    print("  - Modalities: text, audio")
    print("  - Turn detection: server_vad")

    print("\n" + "-" * 60)
    print("Starting realtime session...")
    print("-" * 60)

    try:
        # Start the realtime session
        session = await runner.run()

        async with session:
            print("\nSession started! Processing events...")
            print("(In a real app, you would send audio from a microphone)")
            print()

            # Send a text message to simulate user input
            # In a real app, you would use session.send_audio() with microphone data
            await session.send_message("What's the weather like in Tokyo?")
            print("User (text): What's the weather like in Tokyo?")

            # Process events from the session
            event_count = 0
            max_events = 50  # Limit for demo purposes

            async for event in session:
                event_count += 1
                event_type = getattr(event, 'type', type(event).__name__)

                try:
                    if event_type == "agent_start":
                        agent_name = getattr(event.agent, 'name', 'Unknown')
                        print(f"\n[Agent Started] {agent_name}")

                    elif event_type == "agent_end":
                        agent_name = getattr(event.agent, 'name', 'Unknown')
                        print(f"\n[Agent Ended] {agent_name}")

                    elif event_type == "tool_start":
                        tool_name = getattr(event.tool, 'name', str(event.tool))
                        print(f"\n[Tool Called] {tool_name}")

                    elif event_type == "tool_end":
                        tool_name = getattr(event.tool, 'name', str(event.tool))
                        output = getattr(event, 'output', '')
                        print(f"[Tool Result] {tool_name}: {output}")

                    elif event_type == "audio" or "Audio" in event_type:
                        # In a real app, you would play this audio
                        # Audio data may be in event.data, event.audio, or elsewhere
                        audio_data = getattr(event, 'data', None) or getattr(event, 'audio', None)
                        if audio_data and hasattr(audio_data, '__len__'):
                            print(f"[Audio Output] Received {len(audio_data)} bytes")
                        else:
                            print(f"[Audio Output] Received audio event ({event_type})")

                    elif event_type == "transcript":
                        # Agent's text response (transcription of audio)
                        transcript = getattr(event, 'transcript', '')
                        print(f"[Transcript] {transcript}")

                    elif event_type == "user_transcript":
                        # User's speech transcribed
                        transcript = getattr(event, 'transcript', '')
                        print(f"[User Said] {transcript}")

                    elif event_type == "handoff":
                        from_agent = getattr(getattr(event, 'from_agent', None), 'name', 'Unknown')
                        to_agent = getattr(getattr(event, 'to_agent', None), 'name', 'Unknown')
                        print(f"\n[Handoff] {from_agent} -> {to_agent}")

                    elif event_type == "error":
                        error = getattr(event, 'error', 'Unknown error')
                        print(f"\n[Error] {error}")
                        break

                    elif event_type == "session_end":
                        print("\n[Session Ended]")
                        break

                    else:
                        # Log other event types for debugging
                        print(f"[Event] {event_type}")

                except AttributeError as e:
                    print(f"[Event] {event_type} (attribute access error: {e})")

                # Exit after processing enough events for demo
                if event_count >= max_events:
                    print("\n[Demo] Reached event limit, ending session...")
                    break

    except Exception as e:
        print(f"\nSession error: {e}")
        print("Note: Realtime API requires a valid OpenAI API key with realtime access")

    print("\n" + "=" * 60)
    print("Realtime demo completed!")
    print("OpenTelemetry spans captured for:")
    print("  - Speech synthesis (SpeechSpanData)")
    print("  - Transcription (TranscriptionSpanData)")
    print("  - Speech groups (SpeechGroupSpanData)")
    print("  - Tool calls and agent events")
    print("=" * 60)


async def run_text_only_demo():
    """
    Run a text-only realtime session demonstration.

    This demo uses text-only modalities (no audio), making it simpler to run
    and doesn't require the [voice] extras. It demonstrates:
    1. Text-based realtime conversations
    2. Tool calling in realtime sessions
    3. OpenTelemetry span generation for realtime events
    """
    try:
        from agents import function_tool
        from agents.realtime import RealtimeAgent, RealtimeRunner
    except ImportError:
        print("Error: openai-agents is required for this example")
        print("Install with: pip install openai-agents")
        return

    print("=" * 60)
    print("OpenAI Agents SDK - Realtime Text-Only Demo")
    print("=" * 60)

    # Define tools for the realtime agent
    @function_tool
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        weather_data = {
            "new york": "Partly cloudy, 65°F",
            "london": "Rainy, 52°F",
            "tokyo": "Sunny, 78°F",
            "paris": "Cloudy, 58°F",
            "san francisco": "Foggy, 55°F",
        }
        city_lower = city.lower()
        if city_lower in weather_data:
            return f"The weather in {city} is {weather_data[city_lower]}"
        return f"Weather data not available for {city}"

    @function_tool
    def calculate(expression: str) -> str:
        """Evaluate a simple math expression."""
        try:
            # Only allow safe math operations
            allowed = set('0123456789+-*/.() ')
            if all(c in allowed for c in expression):
                result = eval(expression)
                return f"Result: {result}"
            return "Invalid expression"
        except Exception as e:
            return f"Error: {e}"

    # Create a text-only realtime agent
    assistant = RealtimeAgent(
        name="Text Assistant",
        instructions="""You are a helpful text-based assistant. You can:
- Check the weather in various cities using the get_weather tool
- Perform calculations using the calculate tool

Keep your responses concise and helpful.""",
        tools=[get_weather, calculate],
    )

    print("\nCreated Text Assistant with tools:")
    print("  - get_weather: Get current weather for a city")
    print("  - calculate: Evaluate math expressions")

    # Configure for text-only mode (no audio)
    runner = RealtimeRunner(
        starting_agent=assistant,
        config={
            "model_settings": {
                "model_name": "gpt-4o-realtime-preview",
                "modalities": ["text"],  # Text-only, no audio
            }
        },
    )

    print("\nRunner configured with:")
    print("  - Model: gpt-4o-realtime-preview")
    print("  - Modalities: text (no audio)")

    print("\n" + "-" * 60)
    print("Starting text-only realtime session...")
    print("-" * 60)

    # Messages to send during the demo
    demo_messages = [
        "What's the weather like in San Francisco?",
        "Can you calculate 15 * 7 + 23?",
    ]

    try:
        session = await runner.run()

        async with session:
            print("\nSession started! Processing events...")

            message_index = 0
            event_count = 0
            max_events = 100
            agent_ended = False

            # Send first message
            if demo_messages:
                msg = demo_messages[message_index]
                await session.send_message(msg)
                print(f"\n>>> User: {msg}")
                message_index += 1

            async for event in session:
                event_count += 1
                event_type = getattr(event, 'type', type(event).__name__)

                try:
                    if event_type == "agent_start":
                        agent_name = getattr(event.agent, 'name', 'Unknown')
                        print(f"\n[Agent Started] {agent_name}")

                    elif event_type == "agent_end":
                        agent_name = getattr(event.agent, 'name', 'Unknown')
                        print(f"[Agent Ended] {agent_name}")
                        agent_ended = True

                        # Send next message after agent ends, if we have more
                        if message_index < len(demo_messages):
                            await asyncio.sleep(0.5)  # Brief pause
                            msg = demo_messages[message_index]
                            await session.send_message(msg)
                            print(f"\n>>> User: {msg}")
                            message_index += 1
                            agent_ended = False
                        elif agent_ended:
                            print("\n[Demo] All messages sent, ending session...")
                            break

                    elif event_type == "tool_start":
                        tool_name = getattr(event.tool, 'name', str(event.tool))
                        print(f"  [Tool Called] {tool_name}")

                    elif event_type == "tool_end":
                        tool_name = getattr(event.tool, 'name', str(event.tool))
                        output = getattr(event, 'output', '')
                        print(f"  [Tool Result] {tool_name}: {output}")

                    elif event_type == "history_added":
                        # New item added to conversation history
                        item = getattr(event, 'item', None)
                        if item:
                            role = getattr(item, 'role', None)
                            if role == 'assistant':
                                # Extract text content from assistant message
                                content = getattr(item, 'content', [])
                                for c in content:
                                    if hasattr(c, 'text') and c.text:
                                        print(f"\n<<< Assistant: {c.text}")

                    elif event_type == "error":
                        error = getattr(event, 'error', 'Unknown error')
                        print(f"\n[Error] {error}")
                        break

                    elif event_type in ("history_updated", "raw_model_event"):
                        # Skip verbose events
                        pass

                    else:
                        # Log other event types
                        print(f"  [{event_type}]")

                except AttributeError as e:
                    print(f"  [{event_type}] (error: {e})")

                if event_count >= max_events:
                    print("\n[Demo] Reached event limit, ending session...")
                    break

    except Exception as e:
        print(f"\nSession error: {e}")
        print("Note: Realtime API requires a valid OpenAI API key with realtime access")

    print("\n" + "=" * 60)
    print("Text-only realtime demo completed!")
    print("OpenTelemetry spans captured for:")
    print("  - Realtime Session (workflow span)")
    print("  - Agent start/end events")
    print("  - Tool calls and results")
    print("=" * 60)


async def run_handoff_demo():
    """
    Demonstrate agent handoffs in a realtime text session.

    This shows how multiple agents can handle different parts
    of a conversation with seamless handoffs.
    """
    try:
        from agents.realtime import RealtimeAgent, RealtimeRunner, realtime_handoff
    except ImportError:
        print("Error: openai-agents is required for this example")
        return

    print("\n" + "=" * 60)
    print("OpenAI Agents SDK - Realtime Handoff Demo")
    print("=" * 60)

    # Create specialized agents
    weather_agent = RealtimeAgent(
        name="Weather Expert",
        instructions="""You are a weather specialist.
Provide detailed weather information and forecasts.
Keep responses concise.""",
    )

    time_agent = RealtimeAgent(
        name="Time Expert",
        instructions="""You are a time and scheduling specialist.
Help with time-related questions and scheduling.
Keep responses brief.""",
    )

    # Main agent with handoffs
    main_agent = RealtimeAgent(
        name="Main Assistant",
        instructions="""You are the main assistant.
Route weather questions to the Weather Expert.
Route time questions to the Time Expert.
Handle general queries yourself.""",
        handoffs=[
            realtime_handoff(
                weather_agent,
                tool_description="Transfer to weather expert for weather questions"
            ),
            realtime_handoff(
                time_agent,
                tool_description="Transfer to time expert for time questions"
            ),
        ],
    )

    print("\nCreated agent hierarchy:")
    print("  Main Assistant")
    print("    -> Weather Expert (handoff)")
    print("    -> Time Expert (handoff)")

    runner = RealtimeRunner(
        starting_agent=main_agent,
        config={
            "model_settings": {
                "model_name": "gpt-4o-realtime-preview",
                "modalities": ["text"],  # Text-only mode
            }
        },
    )

    print("\nRunner configured with text-only modalities")
    print(f"Starting agent: {runner.starting_agent.name}")

    print("\n" + "-" * 60)
    print("Starting handoff demo session...")
    print("-" * 60)

    try:
        session = await runner.run()

        async with session:
            print("\nSession started!")

            # Send a weather question to trigger handoff
            await session.send_message("What's the weather like in Tokyo?")
            print("\n>>> User: What's the weather like in Tokyo?")

            event_count = 0
            max_events = 80

            async for event in session:
                event_count += 1
                event_type = getattr(event, 'type', type(event).__name__)

                try:
                    if event_type == "agent_start":
                        agent_name = getattr(event.agent, 'name', 'Unknown')
                        print(f"\n[Agent Started] {agent_name}")

                    elif event_type == "agent_end":
                        agent_name = getattr(event.agent, 'name', 'Unknown')
                        print(f"[Agent Ended] {agent_name}")
                        # End after the first full exchange
                        if event_count > 20:
                            break

                    elif event_type == "handoff":
                        from_agent = getattr(getattr(event, 'from_agent', None), 'name', 'Unknown')
                        to_agent = getattr(getattr(event, 'to_agent', None), 'name', 'Unknown')
                        print(f"\n*** [Handoff] {from_agent} -> {to_agent} ***")

                    elif event_type == "tool_start":
                        tool_name = getattr(event.tool, 'name', str(event.tool))
                        print(f"  [Tool Called] {tool_name}")

                    elif event_type == "tool_end":
                        tool_name = getattr(event.tool, 'name', str(event.tool))
                        output = getattr(event, 'output', '')
                        print(f"  [Tool Result] {tool_name}: {output}")

                    elif event_type == "history_added":
                        item = getattr(event, 'item', None)
                        if item:
                            role = getattr(item, 'role', None)
                            if role == 'assistant':
                                content = getattr(item, 'content', [])
                                for c in content:
                                    if hasattr(c, 'text') and c.text:
                                        print(f"\n<<< Assistant: {c.text}")

                    elif event_type == "error":
                        error = getattr(event, 'error', 'Unknown error')
                        print(f"\n[Error] {error}")
                        break

                    elif event_type not in ("history_updated", "raw_model_event"):
                        print(f"  [{event_type}]")

                except AttributeError as e:
                    print(f"  [{event_type}] (error: {e})")

                if event_count >= max_events:
                    break

    except Exception as e:
        print(f"\nSession error: {e}")

    print("\n" + "=" * 60)
    print("Handoff demo completed!")
    print("OpenTelemetry spans captured handoff events between agents")
    print("=" * 60)


def check_requirements():
    """Check if required packages are installed."""
    missing = []

    try:
        import agents  # noqa: F401
    except ImportError:
        missing.append("openai-agents")

    try:
        from agents.realtime import RealtimeAgent  # noqa: F401
    except ImportError:
        missing.append("openai-agents (realtime module)")

    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install openai-agents")
        return False

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("The realtime session will fail without a valid API key")
        return False

    return True


if __name__ == "__main__":
    import sys

    print("OpenAI Agents SDK - Realtime API Demo")
    print("This demonstrates OpenTelemetry instrumentation for realtime sessions")
    print()

    if not check_requirements():
        print("\nExiting due to missing requirements")
        exit(1)

    # Parse command line arguments
    demo_type = sys.argv[1] if len(sys.argv) > 1 else "text"

    if demo_type == "text":
        print("Running: Text-only realtime demo")
        print("(Use 'voice' argument for audio demo, 'handoff' for handoff demo)")
        print()
        asyncio.run(run_text_only_demo())
    elif demo_type == "voice":
        print("Running: Voice/audio realtime demo")
        print("(Requires openai-agents[voice] extras for full audio support)")
        print()
        asyncio.run(run_realtime_demo())
    elif demo_type == "handoff":
        print("Running: Agent handoff demo")
        print()
        asyncio.run(run_handoff_demo())
    else:
        print(f"Unknown demo type: {demo_type}")
        print("Available options: text, voice, handoff")
        exit(1)
