"""
Beginner's Guide to LLM Tracing with OpenLLMetry
=================================================

This example demonstrates how to trace LLM calls step by step.
OpenLLMetry extends OpenTelemetry to capture details about LLM
requests (tokens, model, prompts, responses) automatically.

What you need:
    pip install traceloop-sdk openai

How it works:
    1. Initialize Traceloop SDK — this auto-instruments LLM libraries
    2. Make regular LLM API calls — tracing happens automatically
    3. View traces in Traceloop dashboard or your OTLP backend

Run:
    export TRACELOOP_API_KEY="your-key"
    python beginners_guide_llm_tracing.py
"""

import os

# ---------------------------------------------------------------------------
# Step 1: Import and initialize Traceloop
# ---------------------------------------------------------------------------
# Traceloop initialization MUST happen before any LLM library imports.
# It auto-instruments OpenAI, Anthropic, and other supported libraries.
# The decorator @task automatically creates OpenTelemetry spans.

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task

Traceloop.init(
    app_name="beginner-llm-guide",
    disable_batch=True,  # Send traces immediately (good for learning)
)

# ---------------------------------------------------------------------------
# Step 2: Import your LLM library (AFTER Traceloop.init)
# ---------------------------------------------------------------------------
# Traceloop auto-instruments this import, wrapping API calls
# with OpenTelemetry spans that capture model, tokens, and timing.

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-..."))


# ---------------------------------------------------------------------------
# Step 3: Define your LLM call with the @task decorator
# ---------------------------------------------------------------------------
# @task creates a parent span. All LLM calls inside it become child spans.
# This gives you a tree view: task → llm_call → token_usage

@task(name="ask_openai")
def ask_llm(question: str) -> str:
    """Ask the LLM a question and return its response."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        max_tokens=100,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Step 4: Run your traced LLM call
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    answer = ask_llm("What is LLM observability in one sentence?")
    print(f"LLM response: {answer}")

    # -----------------------------------------------------------------------
    # Step 5: View your traces
    # -----------------------------------------------------------------------
    # Your trace appears in the Traceloop dashboard at:
    #   https://app.traceloop.com/traces
    #
    # Each trace shows:
    #   - Model used (gpt-4o-mini)
    #   - Tokens consumed (prompt + completion)
    #   - Latency
    #   - Full prompt and response
    #   - Any errors

    print("\nTrace sent! View it at: https://app.traceloop.com/traces")
    print("(Make sure TRACELOOP_API_KEY is set in your environment)")

    # Optional: flush all pending traces before exit
    Traceloop.flush()

# ---------------------------------------------------------------------------
# Next Steps:
#   1. Replace "gpt-4o-mini" with any supported model
#   2. Add @workflow decorator for multi-step agents
#   3. Try @agent to group related tasks
#   4. Export traces to Grafana/Prometheus via OTLP
#   5. Add custom attributes to spans with @task(name="...")
#
# Supported libraries (auto-instrumented):
#   OpenAI, Anthropic, Cohere, Gemini, Mistral, Groq, Bedrock,
#   LangChain, LlamaIndex, Haystack, CrewAI, Ollama, and more.
# ---------------------------------------------------------------------------
