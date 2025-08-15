#!/usr/bin/env python3
"""
Test baseline behavior on main branch - simple handoff demo.
"""

import asyncio
import os
from dotenv import load_dotenv
from agents import Agent, function_tool, Runner
from traceloop.sdk import Traceloop

# Load environment variables from .env file
load_dotenv()

# Initialize Traceloop with OpenAI instrumentation
from traceloop.sdk.instruments import Instruments

Traceloop.init(
    app_name="baseline-test", 
    disable_batch=True,
    instruments={Instruments.OPENAI, Instruments.OPENAI_AGENTS}
)
print("✅ Traceloop initialized - main branch baseline test")

# Simple tools for the Analytics Agent
@function_tool
async def analyze_data(request: str) -> str:
    """Analyze data based on the request."""
    await asyncio.sleep(0.1)
    return f"✅ Analysis complete: {request}"

@function_tool 
async def generate_report(analysis: str) -> str:
    """Generate a report from the analysis."""
    await asyncio.sleep(0.1)  
    return f"📊 Report generated: {analysis}"

async def demo():
    """Run the baseline test on main branch."""
    
    # Create Analytics Agent (handoff target)
    analytics_agent = Agent(
        name="Analytics Agent",
        instructions="You analyze data and generate reports using your tools.",
        model="gpt-4o", 
        tools=[analyze_data, generate_report]
    )
    
    # Create Router Agent (handoff initiator) 
    router_agent = Agent(
        name="Data Router",
        instructions="You route requests to the Analytics Agent specialist.",
        model="gpt-4o",
        handoffs=[analytics_agent]
    )
    
    print("\n🚀 Starting MAIN BRANCH baseline test...")
    print("📊 This should show the handoff hierarchy bug")
    
    # Run the workflow
    query = "Please analyze our Q4 sales data"
    messages = [{"role": "user", "content": query}]
    
    runner = Runner().run_streamed(starting_agent=router_agent, input=messages)
    
    async for event in runner.stream_events():
        if event.type == "run_item_stream_event":
            if "handoff" in event.name.lower():
                print(f"🔄 {event.name}")
            elif "tool" in event.name.lower():  
                print(f"🔧 {event.name}")
    
    print("\n✅ Main branch test complete!")
    print("📊 Expected issue: Analytics Agent appears as ROOT span, not child")
    print("🔗 View spans at: https://app.traceloop.com/")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY environment variable")
        exit(1)
    
    print("🎯 Main Branch Baseline Test")
    print("=" * 40)
    asyncio.run(demo())