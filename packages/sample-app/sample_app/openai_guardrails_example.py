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
from pathlib import Path
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from traceloop.sdk import Traceloop

# Initialize Traceloop FIRST - before importing guardrails
Traceloop.init(app_name="openai-guardrails-example", exporter=ConsoleSpanExporter())

try:
    from guardrails import GuardrailsOpenAI, GuardrailTripwireTriggered
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


def main():
    print("=" * 80)
    print("OpenAI Guardrails with OpenLLMetry Tracing")
    print("=" * 80)

    # Create guardrails configuration
    config_path = create_sample_config()
    print(f"\n[Setup] Created guardrails config: {config_path}")

    # Create GuardrailsOpenAI client
    # This wraps the standard OpenAI client with guardrails validation
    client = GuardrailsOpenAI(api_key=os.getenv("OPENAI_API_KEY"), config=config_path)
    print("[Setup] Created GuardrailsOpenAI client\n")

    # Example 1: Chat Completions with Guardrails
    print("\n--- Example 1: Chat Completions ---")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            temperature=2,
        )

        # With guardrails, the response structure might be different
        if hasattr(response, "llm_response"):
            # Guardrails wrapper format
            print(f"Response: {response.llm_response.choices[0].message.content}")
        else:
            # Standard OpenAI format
            print(f"Response: {response.choices[0].message.content}")

    except GuardrailTripwireTriggered as e:
        print(f"[Guardrails] Safety check triggered: {e}")
    except Exception as e:
        print(f"[Error] {type(e).__name__}: {e}")

    # Example 2: Responses API with Guardrails
    print("\n--- Example 2: Responses API ---")
    try:
        response = client.responses.create(
            model="gpt-4.1-nano",
            input="Explain quantum computing in one sentence.",
            temperature=2,
        )

        # Access the output
        if hasattr(response, "llm_response"):
            output = response.llm_response.output_text
        else:
            output = (
                response.output[0].content[0].text if response.output else "No output"
            )

        print(f"Response: {output}")

    except GuardrailTripwireTriggered as e:
        print(f"[Guardrails] Safety check triggered: {e}")
    except Exception as e:
        print(f"[Error] {type(e).__name__}: {e}")

    print("\n" + "=" * 80)
    print("Tracing Complete!")
    print("=" * 80)
    print("\nCheck your Traceloop dashboard to see:")
    print("  - Complete trace hierarchy maintained across guardrails wrapper")
    print("  - All spans properly nested under the same trace")
    print("  - Request/response details captured with guardrails validation")
    print("\nNote: The fix ensures trace context is maintained even when")
    print("      guardrails makes internal OpenAI API calls.")

    # Cleanup
    if config_path.exists():
        config_path.unlink()
        print(f"\n[Cleanup] Removed {config_path}")


if __name__ == "__main__":
    main()
