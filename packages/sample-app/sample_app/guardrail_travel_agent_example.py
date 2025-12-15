import asyncio
import os
from re import A
from openai import AsyncOpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.guardrails.guardrails import guardrail
from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop
from sample_app.agents.travel_agent_example import run_travel_query


Traceloop.init(
    app_name="travel-agent-example"
)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required. Please set it before running this example.")

client = AsyncOpenAI(api_key=api_key)


# Custom callback to handle PII detection in travel agent responses
def handle_pii_detection(evaluator_result, original_result):
    """
    Custom handler for PII detection in travel agent responses.

    Args:
        evaluator_result: The evaluation result with 'success' and 'reason' fields
        original_result: The original AI response dict (e.g., {"text": "..."})

    Returns:
        Either the original result dict or a sanitized version
    """
    if not evaluator_result.success:
        print("NOMI - handle_pii_detection - evaluator_result:", evaluator_result)
        # PII was detected - return a warning message
        return "⚠️ I detected that my response might contain personally identifiable information (PII). For your privacy and security, I cannot share this information directly. Please contact our travel support team through secure channels for personalized travel details. \n\nReason: {evaluator_result.reason or 'PII detected in response'}"
    return original_result


@guardrail(
    evaluator=EvaluatorMadeByTraceloop.pii_detector(probability_threshold=0.8),
    on_evaluation_complete=handle_pii_detection
)
async def travel_chat_session():
    """Run an interactive travel agent chat session."""
    print("✈️ Welcome to the AI Travel Agent")
    print("=" * 50)
    print("This travel agent is protected by PII detection guardrails.")
    print("Ask me about destinations, travel tips, or general travel advice!")
    print("Type 'quit' to exit the chat.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Safe travels! Goodbye!")
                break

            if not user_input:
                print("Please enter your travel question.")
                continue

            print("\n🤖 Processing your request through the travel agent AI...\n")

            # Get the travel agent's response with PII detection guardrail
            agent_response = await run_travel_query(user_input, return_response_text=True)
            print("NOMI - agent_response:", agent_response)

            return {"text": agent_response}

        except KeyboardInterrupt:
            print("\n\n👋 Chat session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or type 'quit' to exit.")


async def main():
    """Main function to run the travel agent example."""
    await travel_chat_session()


if __name__ == "__main__":
    asyncio.run(main())
