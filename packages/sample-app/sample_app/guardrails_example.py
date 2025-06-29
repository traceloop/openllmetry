"""
Example usage of the guardrails decorator.

This example shows how to use the guardrails decorator to evaluate content
and make runtime decisions based on the calculated score.
"""

from traceloop.sdk import Traceloop
from traceloop.sdk.guardrails import (
    guardrails,
    get_current_score,
    InputExtractor,
)

Traceloop.init(app_name="guardrails-example")



# def calculate_score(event_data: dict) -> float:
#     """Calculate sentiment score from event data."""
#     print(event_data)
#     return event_data.get("pass", False)


# @guardrails(
#     evaluator_slug="What I Hate",
#     score_calculator=calculate_score,
#     input_schema={
#         "love_only": InputExtractor(source="bannans"),
#         "love_sentence": InputExtractor(source="I love bannans because they are Yellow")
#     }
# )
# def process_user_message(message: str, user_id: str) -> str:
#     """Process user message with sentiment analysis."""
#     # Access the sentiment score within the function
#     sentiment_score = get_current_score()
    
#     print(f"Processing message for user {user_id} with sentiment score: {sentiment_score}")
    
#     if sentiment_score:
#         return "Message approved"
#     else:
#         return "Message rejected"


def main():
    """Main function to demonstrate guardrails usage."""
    # Initialize Traceloop (you would need to set TRACELOOP_API_KEY)
    
    print("Guardrails Example Starting")

    try:
        traceloop_client = Traceloop.get()
        traceloop_client.guardrails.execute_evaluator("What I Hate", {"love_only": "bannans", "love_sentence": "I love bannans because they are Yellow"})
        print("Traceloop initialized successfully")
    except Exception as e:
        print(f"Traceloop initialization failed: {e}")
        print("Continuing with mock data...")
    
    # # Example 1: Sentiment analysis
    # print("\n=== Sentiment Analysis Example ===")
    # try:
    #     result = process_user_message("I love this product! It's amazing!", "user123")
    #     print(f"Result: {result}")
    # except Exception as e:
    #     print(f"Error: {e}")

# def test_guardrails():
#     """Test guardrails decorator."""
#     result = process_user_message("I love this product! It's amazing!", "user123")
#     print(f"Result: {result}")


if __name__ == "__main__":
    main() 