"""Debug test to see actual span hierarchy."""

import pytest
from agents import Agent, Runner


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_debug_span_hierarchy(exporter):
    """Debug test to see what spans we actually get."""
    
    # Create simple agent
    agent = Agent(
        name="Test Agent",
        instructions="You are a test agent.",
        model="gpt-4o",
    )
    
    messages = [{"role": "user", "content": "Hello"}]
    runner = Runner().run_streamed(starting_agent=agent, input=messages)
    
    async for event in runner.stream_events():
        pass  # Process all events
    
    spans = exporter.get_finished_spans()
    
    print(f"\n=== DEBUG: Found {len(spans)} spans ===")
    if spans:
        for span in spans:
            parent_name = span.parent.name if span.parent else "ROOT"
            print(f"- {span.name} (parent: {parent_name})")
            
        print(f"\n=== DEBUG: Span details ===")
        for span in spans:
            print(f"Span: {span.name}")
            print(f"  Context: {span.context}")
            print(f"  Parent: {span.parent}")
            print(f"  Attributes: {dict(span.attributes)}")
            print()
    else:
        print("NO SPANS FOUND!")
        
    # Just to prevent test from failing, add a simple assertion
    assert True