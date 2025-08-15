#!/usr/bin/env python3
"""
Sample OpenAI Agents App with Handoff Hierarchy Tracing

This app demonstrates the fixed handoff hierarchy where handed-off agents
appear as child spans under their parent agents in Traceloop.

Run with: python sample_handoff_app.py
Then view the traces in Traceloop to see the beautiful hierarchy!
"""

from traceloop.sdk import Traceloop
import asyncio
import os
from dotenv import load_dotenv
from agents import Agent, function_tool, Runner

# Load environment variables from .env file
load_dotenv()

# Initialize Traceloop for observability
Traceloop.init(
    app_name="agent-handoff-demo",
    disable_batch=True,  # For immediate trace visibility
    api_endpoint=os.getenv("TRACELOOP_API_ENDPOINT", "https://api.traceloop.com"),
    api_key=os.getenv("TRACELOOP_API_KEY"),  # Set your Traceloop API key
)

print("🔧 Traceloop initialized!")
print("📊 Traces will be sent to Traceloop for visualization")


# ============================================================================
# DATA PROCESSING AGENTS WITH TOOLS
# ============================================================================

@function_tool
async def analyze_user_behavior(data_source: str) -> str:
    """Analyze user behavior patterns from the specified data source."""
    await asyncio.sleep(0.1)  # Simulate processing time
    return f"📈 Analyzed user behavior from {data_source}: Found 85% engagement rate with peak activity at 2-4 PM"


@function_tool
async def generate_insights(analysis_data: str) -> str:
    """Generate actionable insights from the analysis data."""
    await asyncio.sleep(0.1)  # Simulate processing time
    return f"💡 Generated insights: {analysis_data} → Recommend scheduling campaigns during peak hours"


@function_tool
async def create_dashboard(insights_data: str) -> str:
    """Create a visual dashboard from the insights."""
    await asyncio.sleep(0.1)  # Simulate processing time
    return f"📊 Created interactive dashboard: {insights_data} → Dashboard available at /analytics/dashboard"


def create_agents():
    """Create the agent workflow: Data Router → Analytics Specialist."""

    # Analytics Specialist Agent (target of handoff)
    analytics_agent = Agent(
        name="Analytics Specialist",
        instructions="""
        You are an expert data analytics specialist. Your role is to:
        1. Analyze user behavior patterns using your analysis tools
        2. Generate actionable business insights from the data
        3. Create visual dashboards for stakeholders

        Always use your tools in sequence: analyze → generate insights → create dashboard.
        Provide detailed explanations of your findings and recommendations.
        """,
        model="gpt-4o",
        tools=[analyze_user_behavior, generate_insights, create_dashboard],
    )

    # Data Router Agent (initiates handoff)
    router_agent = Agent(
        name="Data Router",
        instructions="""
        You are a data routing coordinator. Your role is to:
        1. Understand user requests for data analysis
        2. Route complex analytics tasks to the Analytics Specialist
        3. Provide context and initial guidance

        When users ask for data analysis, behavior analysis, insights, or dashboards,
        hand off to the Analytics Specialist who has the specialized tools and expertise.
        """,
        model="gpt-4o",
        handoffs=[analytics_agent],
    )

    return router_agent, analytics_agent


# ============================================================================
# SAMPLE WORKFLOW SCENARIOS
# ============================================================================

async def run_workflow_scenario(scenario_name: str, user_query: str):
    """Run a specific workflow scenario and show the results."""

    print(f"\n{'='*80}")
    print(f"🚀 SCENARIO: {scenario_name}")
    print(f"{'='*80}")
    print(f"👤 User Query: {user_query}")
    print("🔄 Expected Flow: Data Router → Analytics Specialist → Tools")
    print("📊 Check Traceloop for the complete trace hierarchy!")
    print("-" * 80)

    router_agent, analytics_agent = create_agents()
    messages = [{"role": "user", "content": user_query}]

    try:
        # Run the workflow - handoff should happen automatically
        runner = Runner().run_streamed(starting_agent=router_agent, input=messages)

        print("🔄 Processing workflow...")
        response_parts = []

        async for event in runner.stream_events():
            if event.type == "run_item_stream_event":
                if "handoff" in event.name.lower():
                    print(f"   ✅ Handoff detected: {event.name}")
                elif "tool" in event.name.lower():
                    print(f"   🔧 Tool execution: {event.name}")
                elif event.name == "message_output_created":
                    # Collect response parts
                    raw_item = event.item.raw_item
                    if hasattr(raw_item, 'content'):
                        for part in raw_item.content:
                            if hasattr(part, 'text'):
                                response_parts.append(part.text)

        print("\n📝 Workflow Results:")
        if response_parts:
            full_response = "".join(response_parts)
            print(f"   {full_response}")

        print(f"\n✅ {scenario_name} completed successfully!")
        print("📊 Check your Traceloop dashboard to see the handoff hierarchy:")
        print("   • Data Router (root span)")
        print("   • ├─ Analytics Specialist (child span)")
        print("   • │  ├─ analyze_user_behavior.tool")
        print("   • │  ├─ generate_insights.tool")
        print("   • │  └─ create_dashboard.tool")

    except Exception as e:
        print(f"❌ Error in {scenario_name}: {e}")


async def main():
    """Main application entry point."""

    print("🎯 OpenAI Agents Handoff Hierarchy Demo")
    print("=" * 80)
    print("This demo shows proper agent handoff tracing where:")
    print("• Data Router (parent) hands off to Analytics Specialist (child)")
    print("• All tool executions are properly nested in the trace hierarchy")
    print("• Single unified trace instead of multiple separate traces")
    print("\n💡 Make sure to set your environment variables:")
    print("   export TRACELOOP_API_KEY='your-traceloop-api-key'")
    print("   export OPENAI_API_KEY='your-openai-api-key'")
    print("\n🔗 View results at: https://app.traceloop.com/")

    # Check required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY environment variable not set!")
        print("   Please set your OpenAI API key: export OPENAI_API_KEY='sk-...'")
        return

    if not os.getenv("TRACELOOP_API_KEY"):
        print("\n⚠️  Warning: TRACELOOP_API_KEY not set - traces won't be sent to Traceloop")
        print("   Set your API key: export TRACELOOP_API_KEY='...'")
        print("   You can still see local trace logs below.")

    # Run different workflow scenarios
    scenarios = [
        ("User Behavior Analysis",
         "Can you analyze our website user behavior from the last month and create a dashboard with insights?"),

        ("E-commerce Analytics",
         "I need analysis of customer purchase patterns and recommendations for improving conversion rates."),

        ("Marketing Campaign Analysis",
         "Please analyze the performance of our recent marketing campaigns and suggest optimizations.")
    ]

    for scenario_name, query in scenarios:
        await run_workflow_scenario(scenario_name, query)

        # Brief pause between scenarios
        print("\n⏱️  Waiting 3 seconds before next scenario...")
        await asyncio.sleep(3)

    print("\n🎉 All scenarios completed!")
    print("📊 Check Traceloop dashboard to see the beautiful handoff hierarchy!")
    print("🔗 https://app.traceloop.com/")

    # Keep the app running briefly to ensure traces are sent
    print("\n⏱️  Allowing time for trace delivery...")
    await asyncio.sleep(5)
    print("✅ Demo complete!")


if __name__ == "__main__":
    """
    Usage:
    1. Set environment variables:
       export OPENAI_API_KEY="your-openai-api-key"
       export TRACELOOP_API_KEY="your-traceloop-api-key"  # Optional

    2. Install dependencies:
       pip install openai-agents traceloop-sdk

    3. Run the demo:
       python sample_handoff_app.py

    4. View traces in Traceloop:
       https://app.traceloop.com/
    """
    asyncio.run(main())
