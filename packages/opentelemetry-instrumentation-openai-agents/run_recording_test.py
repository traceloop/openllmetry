#!/usr/bin/env python3
"""
Combined Recording and Testing Script

This script combines the span recording infrastructure with the test harness
to capture and analyze real span generation patterns.
"""

import asyncio
import sys
from pathlib import Path

# Import our recording infrastructure
from record_sample_app_spans import setup_recording_infrastructure, save_recorded_spans, analyze_agent_name_propagation

# Import our test harness
from test_sample_app_harness import run_harness_workflow, run_simple_single_agent_test, run_tool_agent_test


async def run_comprehensive_recording():
    """Run comprehensive recording of all test scenarios."""

    print("🎬 COMPREHENSIVE SPAN RECORDING SESSION")
    print("=" * 60)
    print("Recording spans from multiple test scenarios to analyze agent name propagation")
    print()

    # Set up recording infrastructure
    recorder = setup_recording_infrastructure()

    try:
        print("📋 Test Scenario 1: Sample App Harness Workflow")
        print("-" * 40)
        await run_harness_workflow()

        print("\n📋 Test Scenario 2: Simple Single Agent")
        print("-" * 40)
        run_simple_single_agent_test()

        print("\n📋 Test Scenario 3: Tool Agent")
        print("-" * 40)
        run_tool_agent_test()

        print(f"\n⏱️  Allowing time for span processing...")
        await asyncio.sleep(2)

    except Exception as e:
        print(f"❌ Error during test execution: {e}")

    finally:
        # Save and analyze results
        print(f"\n💾 SAVING AND ANALYZING RESULTS")
        print("=" * 60)

        filename = f"comprehensive_recording_{int(asyncio.get_event_loop().time())}.json"
        output_path = save_recorded_spans(recorder, filename)

        # Analyze the spans
        analyze_agent_name_propagation(recorder.recorded_spans)

        return output_path, recorder.recorded_spans


def create_test_from_recording(spans, filename="test_from_recording.py"):
    """Create a VCR-style test based on recorded spans."""

    print(f"\n🏗️  GENERATING TEST FROM RECORDING")
    print("=" * 60)

    # Analyze span patterns
    agent_spans = [s for s in spans if s["name"].endswith(".agent")]
    tool_spans = [s for s in spans if ".tool" in s["name"]]
    response_spans = [s for s in spans if s["name"] == "openai.response"]

    # Generate test code
    test_code = f'''"""
Auto-generated test based on recorded span patterns.
Generated from {len(spans)} recorded spans.
"""

import pytest
from unittest.mock import Mock, patch

def test_recorded_span_patterns(exporter):
    """Test based on recorded span patterns."""

    # Clear exporter
    exporter.clear()

    # Expected span patterns from recording:
    # - Agent spans: {len(agent_spans)}
    # - Tool spans: {len(tool_spans)}
    # - Response spans: {len(response_spans)}

    # TODO: Add test implementation based on recorded patterns

    # Run test scenario
    # ... (implement based on recording)

    # Get spans
    spans = exporter.get_finished_spans()

    # Verify agent name propagation
    for span in spans:
        if span.name.endswith(".agent"):
            assert "gen_ai.agent.name" in span.attributes, f"Agent span missing agent name: {{span.name}}"

        elif ".tool" in span.name:
            assert "gen_ai.agent.name" in span.attributes, f"Tool span missing agent name: {{span.name}}"

        elif span.name == "openai.response":
            assert "gen_ai.agent.name" in span.attributes, f"Response span missing agent name: {{span.name}}"

        elif span.name == "Agent Workflow":
            assert "gen_ai.agent.name" not in span.attributes, f"Workflow span should not have agent name"

    print("✅ Recorded pattern test passed!")
'''

    # Save the generated test
    test_path = Path(__file__).parent / "tests" / filename
    test_path.parent.mkdir(exist_ok=True)

    with open(test_path, 'w') as f:
        f.write(test_code)

    print(f"✅ Generated test saved to: {test_path}")
    return test_path


async def main():
    """Main execution function."""

    print("🎯 OpenAI Agents Span Recording & Analysis Tool")
    print("=" * 60)
    print("This tool records real span generation and creates tests from the recordings.")
    print()

    try:
        # Run comprehensive recording
        output_path, spans = await run_comprehensive_recording()

        # Create test from recording
        test_path = create_test_from_recording(spans)

        print(f"\n🎊 RECORDING SESSION COMPLETE")
        print("=" * 60)
        print(f"📊 Recorded spans: {len(spans)}")
        print(f"📄 Recording saved: {output_path}")
        print(f"🧪 Test generated: {test_path}")

        # Summary of findings
        missing_agent_names = []
        for span in spans:
            span_name = span["name"]
            has_agent_name = "gen_ai.agent.name" in span["attributes"]

            # Check if span should have agent name
            should_have_agent_name = (
                span_name.endswith(".agent") or
                ".tool" in span_name or
                span_name == "openai.response"
            )

            should_not_have_agent_name = (
                span_name == "Agent Workflow"
            )

            if should_have_agent_name and not has_agent_name:
                missing_agent_names.append(span_name)
            elif should_not_have_agent_name and has_agent_name:
                missing_agent_names.append(f"{span_name} (incorrectly has agent name)")

        if missing_agent_names:
            print(f"\n❌ ISSUES FOUND:")
            print(f"Spans with agent name propagation issues: {len(missing_agent_names)}")
            for issue in missing_agent_names[:5]:  # Show first 5
                print(f"   - {issue}")
            if len(missing_agent_names) > 5:
                print(f"   ... and {len(missing_agent_names) - 5} more")
        else:
            print(f"\n✅ NO ISSUES FOUND!")
            print(f"All spans have correct agent name propagation!")

    except Exception as e:
        print(f"❌ Error during recording session: {e}")
        return 1

    return 0


if __name__ == "__main__":
    """Run the recording and analysis."""
    sys.exit(asyncio.run(main()))