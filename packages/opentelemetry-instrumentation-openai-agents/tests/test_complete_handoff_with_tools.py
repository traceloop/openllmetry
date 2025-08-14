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
    
    print(f"\n🚀 Starting Data Router → Analytics Workflow")
    print(f"📝 Query: {query}")
    print("=" * 80)
    
    # Run the main Data Router agent which should handoff to Analytics
    router_runner = Runner().run_streamed(starting_agent=router_agent, input=messages)
    
    handoff_occurred = False
    async for event in router_runner.stream_events():
        if event.type == "run_item_stream_event":
            if "handoff" in event.name.lower():
                handoff_occurred = True
                print(f"🔄 HANDOFF DETECTED: {event.name}")
            elif "tool" in event.name.lower():
                print(f"🔧 TOOL EXECUTION: {event.name}")
    
    spans = exporter.get_finished_spans()
    non_rest_spans = [span for span in spans if not span.name.endswith("v1/responses")]
    
    # Sort spans by start time for waterfall visualization
    sorted_spans = sorted(non_rest_spans, key=lambda s: s.start_time)
    
    print(f"\n🎯 COMPLETE WORKFLOW TRACE")
    print(f"📊 Total spans: {len(sorted_spans)}")
    print("=" * 80)
    
    # Group by trace ID
    from collections import defaultdict
    traces = defaultdict(list)
    for span in sorted_spans:
        trace_id = span.get_span_context().trace_id
        traces[trace_id].append(span)
    
    for i, (trace_id, trace_spans) in enumerate(traces.items(), 1):
        print(f"\n🔗 TRACE {i}: {trace_id}")
        print(f"   📈 Contains {len(trace_spans)} spans")
        print("-" * 60)
        
        # Build hierarchy
        span_tree = {}
        root_spans = []
        
        for span in trace_spans:
            span_tree[span.context.span_id] = {
                "span": span,
                "children": [],
                "parent_id": span.parent.span_id if span.parent else None
            }
            if span.parent is None:
                root_spans.append(span)
        
        # Build parent-child relationships
        for span_id, span_info in span_tree.items():
            if span_info["parent_id"]:
                parent = span_tree.get(span_info["parent_id"])
                if parent:
                    parent["children"].append(span_info)
        
        # Print hierarchy like the Distillery/GenEdit example
        def print_span_tree(span_info, level=0):
            span = span_info["span"]
            indent = "  " * level
            duration = (span.end_time - span.start_time) / 1_000_000
            
            # Get span type icon
            if "agent" in span.name.lower():
                icon = "🤖" if level == 0 else "  🤖"
                span_type = "Agent"
            elif "tool" in span.name.lower():
                icon = "  🔧"
                span_type = "Tool"
            elif "generation" in span.name.lower():
                icon = "  🔵"
                span_type = "Generation"
            elif "handoff" in span.name.lower():
                icon = "  🔄"
                span_type = "Handoff"
            else:
                icon = "  📋"
                span_type = "Other"
            
            status = "(ROOT)" if level == 0 else "(CHILD)"
            
            print(f"{indent}{icon} {span.name} {status}")
            print(f"{indent}   ⏱️  Duration: {duration:.2f}ms")
            print(f"{indent}   🏷️  Type: {span_type}")
            
            if span.parent:
                print(f"{indent}   👤 Parent: {span.parent.span_id} ✅")
            else:
                print(f"{indent}   👤 Parent: None (ROOT)")
            
            # Show key attributes
            agent_name = span.attributes.get("gen_ai.agent.name")
            if agent_name:
                print(f"{indent}   🎯 Agent: {agent_name}")
            
            tool_name = None
            for attr_key in span.attributes.keys():
                if "tool.name" in attr_key:
                    tool_name = span.attributes[attr_key]
                    break
            if tool_name:
                print(f"{indent}   🔧 Tool: {tool_name}")
            
            print()
            
            for child in span_info["children"]:
                print_span_tree(child, level + 1)
        
        # Print each root span and its children
        for root_span in root_spans:
            root_info = span_tree[root_span.context.span_id]
            print_span_tree(root_info, 0)
    
    print("=" * 80)
    print("🎉 WORKFLOW ANALYSIS COMPLETE!")
    
    # Verify the expected structure like Distillery/GenEdit
    agent_spans = [s for s in non_rest_spans if "agent" in s.name.lower()]
    tool_spans = [s for s in non_rest_spans if "tool" in s.name.lower()]
    root_spans = [s for s in non_rest_spans if s.parent is None]
    child_spans = [s for s in non_rest_spans if s.parent is not None]
    
    print(f"📊 FINAL SUMMARY:")
    print(f"   • Total traces: {len(traces)}")
    print(f"   • Agent spans: {len(agent_spans)}")
    print(f"   • Tool spans: {len(tool_spans)}")
    print(f"   • Root spans: {len(root_spans)}")
    print(f"   • Child spans: {len(child_spans)}")
    print(f"   • Handoff occurred: {handoff_occurred}")
    
    # Assertions for proper workflow
    assert handoff_occurred, "Handoff should have occurred"
    assert len(agent_spans) >= 2, "Should have at least Data Router and Analytics agents"
    assert len(tool_spans) >= 1, "Analytics agent should have used tools"
    assert len(root_spans) == 1, "Should have exactly 1 root span (Data Router Agent)"
    
    # Find the specific agents
    router_spans = [s for s in agent_spans if "Data Router" in s.name]
    analytics_spans = [s for s in agent_spans if "Analytics" in s.name]
    
    assert len(router_spans) == 1, "Should have exactly 1 Data Router Agent span"
    assert len(analytics_spans) >= 1, "Should have at least 1 Analytics Agent span"
    
    # Verify hierarchy: Analytics should be child of Data Router
    router_span = router_spans[0]
    analytics_span = analytics_spans[0]
    
    assert router_span.parent is None, "Data Router Agent should be root"
    assert analytics_span.parent is not None, "Analytics Agent should have parent"
    assert analytics_span.parent.span_id == router_span.context.span_id, "Analytics should be child of Data Router"
    
    print(f"\n✅ SUCCESS: Complete handoff workflow with proper hierarchy!")
    print(f"   🤖 {router_span.name} → 🤖 {analytics_span.name}")
    print(f"   🔧 Tools executed: {len(tool_spans)}")