#!/usr/bin/env python3
"""
LLM Tracing 101 — A beginner-friendly guide to LLM observability with Traceloop.

This example shows how to:
1. Initialize Traceloop
2. Make an LLM call with OpenAI
3. See the trace output in your console (no backend needed!)

Setup:
    pip install openai opentelemetry-sdk opentelemetry-api traceloop-sdk
    export OPENAI_API_KEY="sk-..."

Run:
    python llm_tracing_101.py
"""

import os
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from traceloop.sdk import Traceloop


def main():
    # ------------------------------------------------------------------
    # Step 1: Initialize Traceloop with the ConsoleSpanExporter
    # ------------------------------------------------------------------
    # The ConsoleSpanExporter prints traces to your terminal, so you can
    # see exactly what Traceloop captures without configuring any
    # external observability backend (like Jaeger, Datadog, or Honeycomb).
    Traceloop.init(
        app_name="llm-tracing-101",
        exporter=ConsoleSpanExporter(),
    )

    # ------------------------------------------------------------------
    # Step 2: Make an LLM call
    # ------------------------------------------------------------------
    # Traceloop auto-instruments OpenAI, so every call you make will be
    # traced automatically. No decorators or context managers needed.
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print("Making LLM call...")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is observability in 2 sentences?"},
        ],
        max_tokens=100,
    )

    print(f"\nResponse: {response.choices[0].message.content}")

    # ------------------------------------------------------------------
    # Step 3: View the trace
    # ------------------------------------------------------------------
    # After the script finishes, you'll see a JSON trace printed to your
    # terminal. Look for:
    #   - "name": "openai.chat" — the LLM call span
    #   - "attributes" — model, prompt tokens, completion tokens, etc.
    #
    # In production, swap ConsoleSpanExporter for an OTLP exporter to
    # send traces to your observability backend of choice.


if __name__ == "__main__":
    main()
