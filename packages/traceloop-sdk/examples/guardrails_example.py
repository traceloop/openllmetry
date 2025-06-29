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
    InputSchemaMapping,
)


def calculate_sentiment_score(event_data: dict) -> float:
    """Calculate sentiment score from event data."""
    return event_data.get("sentiment_score", 0.5)


def calculate_toxicity_score(event_data: dict) -> float:
    """Calculate toxicity score from event data."""
    return event_data.get("toxicity_score", 0.0)


def calculate_factuality_score(event_data: dict) -> float:
    """Calculate factuality score from event data."""
    return event_data.get("factuality_score", 0.5)


@guardrails(
    evaluator_slug="sentiment-analyzer",
    score_calculator=calculate_sentiment_score,
    input_schema={
        "text": InputExtractor(source="input", key="message"),
        "user_id": InputExtractor(source="input", key="user_id")
    }
)
def process_user_message(message: str, user_id: str) -> str:
    """Process user message with sentiment analysis."""
    # Access the sentiment score within the function
    sentiment_score = get_current_score()
    
    print(f"Processing message for user {user_id} with sentiment score: {sentiment_score}")
    
    if sentiment_score < 0.2:
        # Very negative sentiment - flag for review
        flag_for_human_review(message, user_id, sentiment_score)
        return "Message flagged for review due to negative sentiment"
    elif sentiment_score < 0.4:
        # Negative sentiment - send to moderation queue
        send_to_moderation_queue(message, user_id)
        return "Message sent to moderation queue"
    elif sentiment_score > 0.8:
        # Very positive sentiment - highlight
        highlight_positive_message(message, user_id)
        return "Message highlighted as positive"
    else:
        # Normal sentiment - process normally
        return f"Message processed normally (sentiment: {sentiment_score:.2f})"


@guardrails(
    evaluator_slug="content-moderation",
    score_calculator=calculate_toxicity_score,
    input_schema={
        "content": InputExtractor(source="input", key="text"),
        "author": InputExtractor(source="input", key="author_id")
    }
)
def moderate_content(text: str, author_id: str) -> str:
    """Moderate content for toxicity."""
    toxicity_score = get_current_score()
    
    print(f"Moderating content from author {author_id} with toxicity score: {toxicity_score}")
    
    if toxicity_score > 0.8:
        # High toxicity - reject immediately
        reject_content(text, author_id, toxicity_score)
        raise ValueError(f"Content rejected due to high toxicity (score: {toxicity_score})")
    elif toxicity_score > 0.6:
        # Moderate toxicity - flag for review
        flag_for_review(text, author_id, toxicity_score)
        return "Content flagged for human review"
    elif toxicity_score > 0.4:
        # Low toxicity - add warning
        add_warning(text, author_id)
        return "Content published with warning"
    else:
        # Low toxicity - approve
        approve_content(text, author_id)
        return "Content approved"


@guardrails(
    evaluator_slug="fact-checker",
    score_calculator=calculate_factuality_score,
    input_schema={
        "claim": InputExtractor(source="input", key="statement"),
        "source": InputExtractor(source="input", key="source_url")
    }
)
def fact_check_statement(statement: str, source_url: str) -> dict:
    """Fact-check a statement."""
    factuality_score = get_current_score()
    
    print(f"Fact-checking statement with factuality score: {factuality_score}")
    
    result = {
        "statement": statement,
        "source": source_url,
        "factuality_score": factuality_score,
        "status": "unknown"
    }
    
    if factuality_score > 0.8:
        result["status"] = "verified"
        result["confidence"] = "high"
    elif factuality_score > 0.6:
        result["status"] = "likely_true"
        result["confidence"] = "medium"
    elif factuality_score > 0.4:
        result["status"] = "uncertain"
        result["confidence"] = "low"
    elif factuality_score > 0.2:
        result["status"] = "likely_false"
        result["confidence"] = "medium"
    else:
        result["status"] = "false"
        result["confidence"] = "high"
    
    return result


# Mock functions for demonstration
def flag_for_human_review(message: str, user_id: str, score: float):
    print(f"Flagging message for human review: {message[:50]}... (score: {score})")

def send_to_moderation_queue(message: str, user_id: str):
    print(f"Sending message to moderation queue: {message[:50]}...")

def highlight_positive_message(message: str, user_id: str):
    print(f"Highlighting positive message: {message[:50]}...")

def reject_content(text: str, author_id: str, score: float):
    print(f"Rejecting content from {author_id} (toxicity: {score})")

def flag_for_review(text: str, author_id: str, score: float):
    print(f"Flagging content for review from {author_id} (toxicity: {score})")

def add_warning(text: str, author_id: str):
    print(f"Adding warning to content from {author_id}")

def approve_content(text: str, author_id: str):
    print(f"Approving content from {author_id}")


def main():
    """Main function to demonstrate guardrails usage."""
    # Initialize Traceloop (you would need to set TRACELOOP_API_KEY)
    try:
        Traceloop.init(app_name="guardrails-example")
        print("Traceloop initialized successfully")
    except Exception as e:
        print(f"Traceloop initialization failed: {e}")
        print("Continuing with mock data...")
    
    # Example 1: Sentiment analysis
    print("\n=== Sentiment Analysis Example ===")
    try:
        result = process_user_message("I love this product! It's amazing!", "user123")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Content moderation
    print("\n=== Content Moderation Example ===")
    try:
        result = moderate_content("This is a great article!", "author456")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Fact checking
    print("\n=== Fact Checking Example ===")
    try:
        result = fact_check_statement(
            "The Earth orbits around the Sun",
            "https://example.com/astronomy"
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 