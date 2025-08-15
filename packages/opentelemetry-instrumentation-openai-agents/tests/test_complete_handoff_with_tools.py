"""Test complete handoff workflow with tools like the Distillery/GenEdit example."""

import pytest
import json
from agents import Agent, function_tool, Runner
from opentelemetry.trace import StatusCode
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues


@function_tool
async def analyze_data(data_request: str) -> str:
    """Analyze the requested data patterns."""
    return f"Analyzed data patterns for: {data_request}"


@function_tool
async def process_results(analysis_data: str) -> str:
    """Process the analysis results."""
    return f"Processed results: {analysis_data}"


@function_tool
async def generate_report(processed_data: str) -> str:
    """Generate a final report from the processed data."""
    return f"Generated report: {processed_data}"


@pytest.fixture(scope="session")
def workflow_agents():
    """Create Data Router Agent and Analytics Agent with function tools."""
    
    # Create Analytics agent with data processing tools
    analytics_agent = Agent(
        name="Analytics Agent",
        instructions="You are a data analytics specialist. Use your tools to analyze data, process results, and generate reports.",
        model="gpt-4o",
        tools=[analyze_data, process_results, generate_report],
    )
    
    # Create Data Router agent that can hand off to Analytics
    router_agent = Agent(
        name="Data Router Agent",
        instructions="You handle general requests and route data processing tasks to the Analytics Agent.",
        model="gpt-4o",
        handoffs=[analytics_agent],
    )
    
    return router_agent, analytics_agent


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_router_analytics_complete_workflow(exporter, workflow_agents):
    """Test complete Data Router → Analytics handoff with tool execution."""
    
    router_agent, analytics_agent = workflow_agents
    
    query = "Can you analyze the sales data from last quarter and generate a report?"
    messages = [{"role": "user", "content": query}]
    
    
    # Run the main Data Router agent which should handoff to Analytics
    router_runner = Runner().run_streamed(starting_agent=router_agent, input=messages)
    
    handoff_occurred = False
    async for event in router_runner.stream_events():
        if event.type == "run_item_stream_event":
            if "handoff" in event.name.lower():
                handoff_occurred = True
            elif "tool" in event.name.lower():
                pass
    
    spans = exporter.get_finished_spans()
    non_rest_spans = [span for span in spans if not span.name.endswith("v1/responses")]
    
    
    # Verify the expected structure like Distillery/GenEdit
    agent_spans = [s for s in non_rest_spans if "agent" in s.name.lower()]
    tool_spans = [s for s in non_rest_spans if "tool" in s.name.lower()]
    root_spans = [s for s in non_rest_spans if s.parent is None]
    child_spans = [s for s in non_rest_spans if s.parent is not None]
    
    # Assertions for proper workflow
    assert handoff_occurred, "Handoff should have occurred"
    assert len(agent_spans) >= 2, "Should have at least Data Router and Analytics agents"
    assert len(tool_spans) >= 1, "Analytics agent should have used tools"
    assert len(root_spans) == 1, "Should have exactly 1 root span (Agent Workflow)"
    
    # Find the specific agents, excluding handoff spans from individual agent counts
    router_spans = [s for s in agent_spans if "Data Router" in s.name and ".agent" in s.name]
    analytics_spans = [s for s in agent_spans if "Analytics" in s.name and ".agent" in s.name]
    handoff_spans = [s for s in agent_spans if "handoff" in s.name]
    
    assert len(router_spans) == 1, f"Should have exactly 1 Data Router Agent span, found {len(router_spans)}: {[s.name for s in router_spans]}"
    assert len(analytics_spans) >= 1, f"Should have at least 1 Analytics Agent span, found {len(analytics_spans)}: {[s.name for s in analytics_spans]}"
    
    # Verify hierarchy: All agents should be children of workflow span
    router_span = router_spans[0]
    analytics_span = analytics_spans[0]
    workflow_spans = [s for s in agent_spans if s.name == "Agent Workflow"]
    workflow_span = workflow_spans[0]
    
    assert workflow_span.parent is None, "Agent Workflow should be root"
    assert router_span.parent is not None, "Data Router Agent should have parent"
    assert analytics_span.parent is not None, "Analytics Agent should have parent"
    assert router_span.parent.span_id == workflow_span.context.span_id, "Data Router should be child of Workflow"
    assert analytics_span.parent.span_id == workflow_span.context.span_id, "Analytics should be child of Workflow"