"""
Beginner-friendly example: Basic LLM Tracing with OpenLLMetry

This example demonstrates the minimal setup needed to trace LLM calls
using OpenLLMetry. It shows how to:
1. Initialize the Traceloop SDK
2. Make a traced LLM call
3. View the traces in your observability backend

Prerequisites:
- pip install traceloop-sdk openai
- Set OPENAI_API_KEY environment variable (or replace with your key)
- Optional: Set TRACELOOP_API_KEY to send traces to Traceloop dashboard
"""

import os
from openai import OpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow


def main():
    # Step 1: Initialize Traceloop SDK
    # This sets up OpenTelemetry instrumentation for LLM calls
    # disable_batch=True sends traces immediately (useful for local development)
    # app_name identifies your application in the observability backend
    Traceloop.init(
        app_name="basic_llm_tracing_example",
        disable_batch=True,  # Set to False in production for better performance
    )

    # Step 2: Create your LLM client as usual
    # The OpenLLMetry SDK automatically instruments popular LLM clients
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Step 3: Use the @workflow decorator to trace your LLM workflow
    # This creates a parent span that groups all LLM calls within the function
    @workflow(name="basic_chat_completion")
    def chat_workflow():
        # This call will be automatically traced by OpenLLMetry
        # The trace will capture: model, prompt, response, tokens, latency, etc.
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain OpenTelemetry in one sentence."},
            ],
            max_tokens=100,
        )
        print(f"Response: {response.choices[0].message.content}")
        return response

    # Step 4: Run the workflow
    chat_workflow()

    print("\n✅ Trace complete! Check your observability backend for the trace.")


if __name__ == "__main__":
    main()