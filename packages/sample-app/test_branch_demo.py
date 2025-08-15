#!/usr/bin/env python3
"""
Simple OpenAI Agents Handoff Demo for Traceloop

A minimal example showing the fixed handoff hierarchy.
Run this to see perfect parent-child span relationships in Traceloop!
"""

import asyncio
import os
from agents import Agent, function_tool, Runner
from traceloop.sdk import Traceloop

# Set up API key directly if needed
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️ OPENAI_API_KEY not found in environment. Please set it.")
    print("   You can either:")
    print("   1. Set it in your shell: export OPENAI_API_KEY=sk-your-key")  
    print("   2. Or edit the .env file in this directory")
    exit(1)

# Initialize Traceloop with OpenAI instrumentation
from traceloop.sdk.instruments import Instruments

Traceloop.init(
    app_name="handoff-demo", 
    disable_batch=True,
    # Ensure OpenAI instrumentation is enabled
    instruments={Instruments.OPENAI, Instruments.OPENAI_AGENTS}
)

# Debug: Check which instrumentors are active
from opentelemetry._logs import get_logger
debug_logger = get_logger(__name__)
print("🔍 Checking active OpenTelemetry instrumentors...")

try:
    from opentelemetry.instrumentation.registry import _INSTRUMENTED_LIBRARIES
    print(f"📋 Active instrumentors: {list(_INSTRUMENTED_LIBRARIES.keys())}")
except:
    print("❓ Could not access instrumentation registry")

# Test basic OpenAI call to verify completion spans
print("🧪 Testing basic OpenAI completion to verify spans...")
from openai import OpenAI
test_client = OpenAI()
try:
    test_response = test_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=5
    )
    print(f"✅ Basic OpenAI test successful: {test_response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Basic OpenAI test failed: {e}")
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