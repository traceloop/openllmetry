#!/usr/bin/env python3
"""
Simple OpenAI Agents Handoff Demo for Traceloop

A minimal example showing the fixed handoff hierarchy.
Run this to see perfect parent-child span relationships in Traceloop!
"""

from traceloop.sdk.instruments import Instruments
import asyncio
import os
from dotenv import load_dotenv
from agents import Agent, function_tool, Runner
from traceloop.sdk import Traceloop

# Load environment variables from .env file
load_dotenv()

# Initialize Traceloop with dashboard exporter

Traceloop.init(
    app_name="handoff-demo",
    disable_batch=True,
    # Ensure OpenAI instrumentation is enabled
    instruments={Instruments.OPENAI, Instruments.OPENAI_AGENTS}
)
print("✅ Traceloop initialized - traces will appear in your dashboard!")

# Simple tools for the Analytics Agent


@function_tool
async def analyze_data(request: str) -> str:
    """Analyze data based on the request."""
    await asyncio.sleep(0.2)  # Simulate work
    return f"✅ Analysis complete: {request} → Key metrics identified"


@function_tool
async def generate_report(analysis: str) -> str:
    """Generate a report from the analysis."""
    await asyncio.sleep(0.2)  # Simulate work
    return f"📊 Report generated: {analysis} → Executive summary created"


async def demo():
    """Run the handoff demo."""

    # Create Analytics Agent (handoff target)
    analytics_agent = Agent(
        name="Analytics Agent",
        instructions="You analyze data and generate reports using your tools. Use both tools in sequence.",
        model="gpt-4o",
        tools=[analyze_data, generate_report]
    )

    # Create Router Agent (handoff initiator)
    router_agent = Agent(
        name="Data Router",
        instructions="You route data analysis requests to the Analytics Agent specialist.",
        model="gpt-4o",
        handoffs=[analytics_agent]
    )

    print("\n🚀 Starting handoff workflow...")
    print("📊 Check Traceloop to see the hierarchy!")

    # Run the workflow
    query = "Please analyze our Q4 sales data and generate an executive report"
    messages = [{"role": "user", "content": query}]

    runner = Runner().run_streamed(starting_agent=router_agent, input=messages)

    async for event in runner.stream_events():
        if event.type == "run_item_stream_event":
            if "handoff" in event.name.lower():
                print(f"🔄 {event.name}")
            elif "tool" in event.name.lower():
                print(f"🔧 {event.name}")

    print("\n✅ Demo complete!")
    print("📊 Expected trace hierarchy in Traceloop:")
    print("   🌐 Agent Workflow (parent span - covers entire workflow)")
    print("   ├─ 🤖 Data Router (sibling - covers only routing execution ~2s)")
    print("   ├─ 🔄 Data Router → Analytics Agent.handoff (explicit handoff)")
    print("   └─ 🤖 Analytics Agent (sibling - covers analysis work ~6s)")
    print("      ├─ 🔧 analyze_data.tool")
    print("      └─ 🔧 generate_report.tool")
    print("\n🔗 View at: https://app.traceloop.com/")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY environment variable")
        exit(1)

    print("🎯 OpenAI Agents Handoff Demo")
    print("=" * 40)
    asyncio.run(demo())
