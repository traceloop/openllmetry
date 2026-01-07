"""
OpenAI Agents SDK Realtime Voice Agent Example

This example demonstrates how to use the OpenAI Agents SDK's realtime
voice capabilities with OpenTelemetry instrumentation.

The realtime API enables:
- Real-time voice conversations with AI
- Tool calling during voice sessions
- Agent handoffs in voice context
- Transcription and speech synthesis tracking

Requirements:
- pip install openai-agents[voice]
- OPENAI_API_KEY environment variable

Note: This example uses simulated audio for demonstration purposes.
In a real application, you would connect to a microphone and speaker.
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
        print("Error: openai-agents[voice] is required for this example")
        print("Install with: pip install openai-agents[voice]")
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

                if event.type == "agent_start":
                    print(f"\n[Agent Started] {event.agent.name}")

                elif event.type == "agent_end":
                    print(f"\n[Agent Ended] {event.agent.name}")

                elif event.type == "tool_start":
                    tool_name = getattr(event.tool, 'name', str(event.tool))
                    print(f"\n[Tool Called] {tool_name}")

                elif event.type == "tool_end":
                    tool_name = getattr(event.tool, 'name', str(event.tool))
                    print(f"[Tool Result] {tool_name}: {event.output}")

                elif event.type == "audio":
                    # In a real app, you would play this audio
                    audio_len = len(event.audio) if hasattr(event, 'audio') else 0
                    print(f"[Audio Output] Received {audio_len} bytes")

                elif event.type == "transcript":
                    # Agent's text response (transcription of audio)
                    print(f"[Transcript] {event.transcript}")

                elif event.type == "user_transcript":
                    # User's speech transcribed
                    print(f"[User Said] {event.transcript}")

                elif event.type == "handoff":
                    from_agent = getattr(event.from_agent, 'name', 'Unknown')
                    to_agent = getattr(event.to_agent, 'name', 'Unknown')
                    print(f"\n[Handoff] {from_agent} -> {to_agent}")

                elif event.type == "error":
                    print(f"\n[Error] {event.error}")
                    break

                elif event.type == "session_end":
                    print("\n[Session Ended]")
                    break

                else:
                    # Log other event types for debugging
                    print(f"[Event] {event.type}")

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


async def run_handoff_demo():
    """
    Demonstrate agent handoffs in a realtime voice session.

    This shows how multiple agents can handle different parts
    of a voice conversation with seamless handoffs.
    """
    try:
        from agents.realtime import RealtimeAgent, RealtimeRunner, realtime_handoff
    except ImportError:
        print("Error: openai-agents[voice] is required for this example")
        return

    print("\n" + "=" * 60)
    print("OpenAI Agents SDK - Realtime Handoff Demo")
    print("=" * 60)

    # Create specialized agents
    weather_agent = RealtimeAgent(
        name="Weather Expert",
        instructions="""You are a weather specialist.
Provide detailed weather information and forecasts.
Keep responses conversational for voice.""",
    )

    time_agent = RealtimeAgent(
        name="Time Expert",
        instructions="""You are a time and scheduling specialist.
Help with time-related questions and scheduling.
Keep responses brief for voice.""",
    )

    # Main agent with handoffs
    main_agent = RealtimeAgent(
        name="Main Assistant",
        instructions="""You are the main voice assistant.
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
                "voice": "nova",
                "modalities": ["text"],  # Text-only for simpler demo
            }
        },
    )

    print("\nHandoff demo configured")
    print(f"  Runner starting agent: {runner.starting_agent.name}")
    print("(This demonstrates the handoff tracking in realtime sessions)")


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
        missing.append("openai-agents[voice]")

    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall with: pip install openai-agents[voice]")
        return False

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("The realtime session will fail without a valid API key")
        return False

    return True


if __name__ == "__main__":
    print("OpenAI Agents SDK - Realtime Voice Agent Demo")
    print("This demonstrates OpenTelemetry instrumentation for realtime voice")
    print()

    if not check_requirements():
        print("\nExiting due to missing requirements")
        exit(1)

    # Run the main demo
    asyncio.run(run_realtime_demo())
