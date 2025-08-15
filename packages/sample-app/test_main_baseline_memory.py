#!/usr/bin/env python3
"""
Test baseline behavior on main branch with memory exporter to see exact spans.
"""

import asyncio
import os
from agents import Agent, function_tool, Runner

# Set API key directly if not in environment
if not os.getenv("OPENAI_API_KEY"):
    # For testing, we'll use a dummy key and expect it to fail gracefully
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-testing"
    print("⚠️ Using dummy API key - calls will fail but spans will be created")

# Use Traceloop's memory-like approach by checking spans
from traceloop.sdk import Traceloop
from traceloop.sdk.instruments import Instruments

# Initialize Traceloop but disable export to avoid API key issues
Traceloop.init(
    app_name="baseline-test",
    disable_batch=True,  # This helps us see spans locally
    instruments={Instruments.OPENAI, Instruments.OPENAI_AGENTS}
)

print("✅ Traceloop initialized for main branch baseline test")

# Simple tools
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
    
    print("\n🚀 Starting MAIN BRANCH baseline test with memory exporter...")
    
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
    print("📊 On main branch, we expect to see the handoff hierarchy bug")
    print("🔗 The Analytics Agent should appear as a ROOT span instead of child")
    print("\n💡 To see actual spans, check Traceloop dashboard or use a test with memory exporter")

if __name__ == "__main__":
    print("🎯 Main Branch Baseline Test")
    print("=" * 40)
    asyncio.run(demo())