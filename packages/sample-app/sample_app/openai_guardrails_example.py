"""
OpenAI Guardrails Example with Traceloop SDK

This example demonstrates how to use OpenAI's guardrails library with
OpenLLMetry tracing. The guardrails library wraps the OpenAI client to
provide automatic input/output validation and safety checks.

The trace context is properly maintained across the wrapper's internal calls,
ensuring complete observability of the request flow.
"""

import os
import json
import asyncio
from pathlib import Path
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from traceloop.sdk import Traceloop

# Initialize Traceloop FIRST - before importing guardrails
Traceloop.init(
    app_name="openai-guardrails-example",
    api_endpoint="https://api.traceloop.com",
    disable_batch=True,
    exporter=ConsoleSpanExporter()
)

try:
    from guardrails import GuardrailsAsyncOpenAI, GuardrailTripwireTriggered
except ImportError:
    print("=" * 80)
    print("ERROR: openai-guardrails-python not installed")
    print("=" * 80)
    print("\nTo install:")
    print("  pip install openai-guardrails")
    print("\nDocumentation: https://openai.github.io/openai-guardrails-python/")
    print("=" * 80)
    exit(1)


def create_sample_config():
    """Create a minimal guardrails configuration."""
    config = {
        "version": 1,
        "input": {
            "version": 1,
            "guardrails": [],
        },
        "output": {
            "version": 1,
            "guardrails": [],
        },
    }

    config_path = Path("guardrails_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


async def reproduce_issue():
    """Reproduce the broken trace issue with streaming responses."""
    print("=" * 80)
    print("OpenAI Guardrails with OpenLLMetry Tracing - Streaming Test")
    print("=" * 80)

    # Create guardrails configuration
    config_path = create_sample_config()
    print(f"\n[Setup] Created guardrails config: {config_path}")

    # Initialize GuardrailsAsyncOpenAI (wraps AsyncOpenAI)
    client = GuardrailsAsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        config=config_path
    )
    print("[Setup] Created GuardrailsAsyncOpenAI client\n")

    # Call responses.create() with streaming
    print("\n--- Streaming Responses API ---")
    try:
        stream = await client.responses.create(
            model="gpt-4o",
            input="Hello, how are you?",
            stream=True
        )

        print("Streaming chunks:")
        async for chunk in stream:
            # Extract delta from the chunk
            if hasattr(chunk, 'llm_response') and hasattr(chunk.llm_response, 'delta'):
                print(chunk.llm_response.delta, end='', flush=True)
            elif hasattr(chunk, 'delta'):
                print(chunk.delta, end='', flush=True)
        print()  # New line after streaming

    except GuardrailTripwireTriggered as e:
        print(f"[Guardrails] Safety check triggered: {e}")
    except Exception as e:
        print(f"[Error] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Check console output above for spans")
    print("=" * 80)
    print("\nLook for multiple trace IDs - this indicates broken traces!")
    print("All spans should share the same trace_id for proper aggregation.")

    # Cleanup
    if config_path.exists():
        config_path.unlink()
        print(f"\n[Cleanup] Removed {config_path}")


def main():
    """Main entry point."""
    asyncio.run(reproduce_issue())


if __name__ == "__main__":
    main()
