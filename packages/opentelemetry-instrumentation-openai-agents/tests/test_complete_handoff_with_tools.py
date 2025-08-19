"""Test complete handoff workflow with tools like the Distillery/GenEdit example."""

import pytest
from agents import Agent, function_tool, Runner


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
        instructions=(
            "You are a data analytics specialist. Use your tools to analyze data, "
            "process results, and generate reports."
        ),
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
    agent_spans = [s for s in non_rest_spans if s.name.endswith(".agent")]
    tool_spans = [s for s in non_rest_spans if s.name.endswith(".tool")]
    workflow_spans = [s for s in non_rest_spans if s.name == "Agent Workflow"]
    handoff_spans = [s for s in non_rest_spans if ".handoff" in s.name]
    root_spans = [s for s in non_rest_spans if s.parent is None]

    # Assertions for proper workflow
    assert handoff_occurred, "Handoff should have occurred"
    assert len(workflow_spans) == 1, f"Should have exactly 1 Agent Workflow span, found {len(workflow_spans)}"
    assert len(tool_spans) >= 1, (
        f"Analytics agent should have used tools, found {len(tool_spans)}: "
        f"{[s.name for s in tool_spans]}"
    )
    assert len(root_spans) == 1, (
        f"Should have exactly 1 root span (Agent Workflow), found {len(root_spans)}: "
        f"{[s.name for s in root_spans]}"
    )

    # Find the specific agents - Data Router might not create its own span if it immediately hands off
    router_spans = [s for s in agent_spans if "Data Router Agent" in s.name]
    analytics_spans = [s for s in agent_spans if "Analytics Agent" in s.name]

    # The key requirement is that we have Analytics agent spans and proper handoff
    assert len(analytics_spans) >= 1, (
        f"Should have at least 1 Analytics Agent span, found {len(analytics_spans)}: "
        f"{[s.name for s in analytics_spans]}"
    )
    assert len(handoff_spans) >= 1, (
        f"Should have at least 1 handoff span, found {len(handoff_spans)}: "
        f"{[s.name for s in handoff_spans]}"
    )

    # Verify hierarchy: Analytics agent should be child of workflow span
    analytics_span = analytics_spans[0]
    workflow_span = workflow_spans[0]
    handoff_span = handoff_spans[0]

    assert workflow_span.parent is None, "Agent Workflow should be root"
    assert analytics_span.parent is not None, "Analytics Agent should have parent"
    assert analytics_span.parent.span_id == workflow_span.context.span_id, "Analytics should be child of Workflow"

    # Handoff span should be child of router agent (if it exists) or workflow span
    assert handoff_span.parent is not None, "Handoff span should have parent"
    if router_spans:
        # If router span exists, handoff should be its child
        router_span = router_spans[0]
        assert handoff_span.parent.span_id == router_span.context.span_id, (
            "Handoff should be child of Data Router Agent"
        )
    else:
        # If router span doesn't exist, handoff should be child of workflow
        assert handoff_span.parent.span_id == workflow_span.context.span_id, "Handoff should be child of Workflow"

    # If router span exists, verify its hierarchy too
    if router_spans:
        router_span = router_spans[0]
        assert router_span.parent is not None, "Data Router Agent should have parent"
        assert router_span.parent.span_id == workflow_span.context.span_id, "Data Router should be child of Workflow"
