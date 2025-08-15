#!/usr/bin/env python3

"""Simple test to check OpenAI agents span hierarchy with console output."""

import asyncio
from dotenv import load_dotenv
from agents import Agent, function_tool, Runner
from traceloop.sdk import Traceloop
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

# Load environment variables from .env file
load_dotenv()

# Initialize normally (without console exporter)
Traceloop.init(
    app_name="agents-hierarchy-test",
    disable_batch=True
)

@function_tool
async def analyze_data(query: str) -> str:
    """Analyze Q4 sales data for trends and key performance metrics."""
    print(f"🔧 TOOL: Analyzing data for: {query}")
    return "✅ Analysis complete: Analyze Q4 sales data for trends, key performance metrics, and comparisons with previous quarters. → Key metrics identified"

@function_tool
async def generate_report(analysis: str) -> str:
    """Generate a comprehensive report based on the analysis."""
    print(f"🔧 TOOL: Generating report for: {analysis}")
    return "📊 Report generated successfully"

def main():
    """Test OpenAI agents span hierarchy."""
    print("Testing OpenAI agents span hierarchy...")
    print("Expected hierarchy:")
    print("  Agent workflow (root)")
    print("  ├── Analytics Agent (agent)")
    print("  │   ├── analyze_data (tool)")
    print("  │   └── generate_report (tool)")
    print("="*60)
    
    # Create agent with tools
    analytics_agent = Agent(
        name="Analytics Agent",
        instructions="You analyze data and generate reports using your tools in sequence.",
        model="gpt-4o",
        tools=[analyze_data, generate_report],
    )
    
    # Simple query that should use both tools
    query = "Analyze Q4 sales data for trends, key performance metrics, and comparisons with previous quarters."
    
    print(f"Running query: {query}")
    print("-"*60)
    
    # Run the agent
    result = Runner.run_sync(analytics_agent, query)
    
    print("-"*60)
    print("✅ Execution complete. Check console output above for span hierarchy.")
    print("Looking for tool spans nested under agent spans...")

if __name__ == "__main__":
    main()