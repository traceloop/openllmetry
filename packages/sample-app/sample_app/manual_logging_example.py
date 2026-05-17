"""
Manual logging example using OpenLLMetry tracing.

This example demonstrates how to manually report LLM calls using the
track_llm_call context manager from traceloop.sdk.tracing.manual.
"""

import os
import openai
from dotenv import load_dotenv
from traceloop.sdk import Traceloop
from traceloop.sdk.tracing.manual import LLMMessage, LLMUsage, track_llm_call


def main():
    # Load environment variables
    load_dotenv()

    # Initialize Traceloop
    Traceloop.init(
        app_name="manual-logging-example",
        disable_batch=True,
        instruments={}
    )

    # Initialize OpenAI client
    openai_client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print("Starting manual logging example...")

    # Example 1: Basic LLM call with manual reporting
    print("\n=== Example 1: Basic LLM Call ===")
    with track_llm_call(vendor="openai", type="chat") as span:
        # Report the request
        messages = [
            LLMMessage(role="user", content="Tell me a joke about opentelemetry")
        ]
        span.report_request(
            model="gpt-4o",
            messages=messages,
        )

        # Make the actual API call
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )

        # Report the response
        response_messages = [choice.message.content for choice in response.choices]
        span.report_response(response.model, response_messages)

        # Report usage metrics
        if response.usage:
            span.report_usage(
                LLMUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
            )

        print(f"Response: {response_messages[0]}")

    # Example 2: Second independent LLM call
    print("\n=== Example 2: Second LLM Call ===")
    with track_llm_call(vendor="openai", type="chat") as span:
        # Report the request
        messages = [
            LLMMessage(role="user", content="What is machine learning?")
        ]
        span.report_request(
            model="gpt-4o",
            messages=messages,
        )

        # Make the actual API call
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "What is machine learning?"}
            ],
        )

        # Report the response
        response_messages = [choice.message.content for choice in response.choices]
        span.report_response(response.model, response_messages)

        # Report usage metrics
        if response.usage:
            span.report_usage(
                LLMUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
            )

        print(f"Response: {response_messages[0]}")

    print("\nManual logging example completed!")


if __name__ == "__main__":
    main()
