"""
Guardrail wrapper for the existing travel agent example.
This demonstrates how to add PII detection guardrails to an existing agentic system.
"""
import asyncio
import sys
from pathlib import Path

# Add the agents directory to the path
agents_dir = Path(__file__).parent / "agents"
sys.path.insert(0, str(agents_dir))

from traceloop.sdk.guardrails.guardrails import guardrail  # noqa: E402
from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop  # noqa: E402

# Import the travel agent function
try:
    from travel_agent_example import run_travel_query
except ImportError:
    print("Error: Could not import travel_agent_example.")
    print(f"Make sure {agents_dir}/travel_agent_example.py exists")
    sys.exit(1)


# Custom callback to handle PII detection in travel agent responses
def handle_pii_detection(evaluator_result, original_result):
    """
    Custom handler for PII detection in travel agent itineraries.

    Args:
        evaluator_result: The evaluation result with 'success' and 'reason' fields
        original_result: The original travel itinerary response dict

    Returns:
        Either the original result dict or a sanitized warning
    """
    # Get captured stdout
    captured_stdout = original_result.get("_captured_stdout", "")

    if not evaluator_result.success:
        # PII was detected - don't display the output, return warning
        return {
            "text": (
                "⚠️ PRIVACY ALERT: The generated travel itinerary contains personally "
                "identifiable information (PII) that could compromise your privacy.\n\n"
                "For your security, we cannot display this itinerary. Please:\n"
                "1. Contact our support team through secure channels\n"
                "2. Request a generic itinerary without personal details\n"
                "3. Avoid sharing sensitive information in your travel requests\n\n"
                f"Detection reason: {evaluator_result.reason or 'PII detected in response'}"
            )
        }
    else:
        # Guardrail passed - now safe to display the captured output
        if captured_stdout:
            print(captured_stdout, end="")

        # Remove internal field before returning
        result = original_result.copy()
        result.pop("_captured_stdout", None)
        return result


@guardrail(
    evaluator=EvaluatorMadeByTraceloop.pii_detector(probability_threshold=0.7),
    on_evaluation_complete=handle_pii_detection
)
async def guarded_travel_agent(query: str) -> dict:
    """
    Wrapper around the travel agent that adds PII detection guardrails.

    This function:
    1. Runs the full travel agent flow (tools, API calls, itinerary generation)
    2. Gets the final response text from the agent
    3. Runs PII detection on the complete output
    4. Returns sanitized response if PII is detected

    Args:
        query: User's travel planning request

    Returns:
        Dict with 'text' field containing the travel itinerary or privacy warning
    """
    import io

    # Capture stdout to prevent streaming output before guardrail check
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    try:
        # Run the travel agent and get the response text
        # return_response_text=True makes it return the agent's text instead of tool_calls
        response_text = await run_travel_query(query, return_response_text=True)

        # Return dict with 'text' field as required by pii_detector
        return {"text": response_text, "_captured_stdout": captured_output.getvalue()}

    finally:
        # Restore stdout
        sys.stdout = old_stdout


async def main():
    """
    Main function demonstrating guardrails with the travel agent.
    """
    print("=" * 80)
    print("Travel Agent with PII Detection Guardrails")
    print("=" * 80)
    print("This example wraps the existing travel agent with PII detection.")
    print("If the generated itinerary contains PII, it will be blocked.")
    print("=" * 80)
    print()

    # Test query - should pass as it's a general request
    test_query = "Plan a 5-day moderate trip to Paris for couples interested in food and museums."

    print(f"Query: {test_query}\n")

    try:
        # Run the guarded travel agent
        result = await guarded_travel_agent(test_query)

        # Display the result
        print("\n" + "=" * 80)
        print("FINAL RESPONSE (after PII check):")
        print("=" * 80)
        response_text = result.get("text", "")

        # Show first 1000 chars to avoid cluttering terminal
        if len(response_text) > 1000:
            print(response_text[:1000] + "...\n[Response truncated for display]")
        else:
            print(response_text)

        print("=" * 80)
        print("✅ Query completed with PII guardrail check")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ Guardrail demo completed!")
    print("=" * 80)
    print("\nKey features demonstrated:")
    print("  - Existing travel agent with 6 tools and agentic workflows")
    print("  - PII detection guardrail on final output")
    print("  - Custom callback for handling PII violations")
    print("  - Seamless integration with existing code")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
