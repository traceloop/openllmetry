"""
Beginner-friendly example: LLM tracing with Aleph Alpha and OpenLLMetry
========================================================================

This script shows, step by step, how to:
  1. Initialise the Traceloop SDK (which sets up the OpenTelemetry tracer).
  2. Instrument the Aleph Alpha client so every completion call is traced
     automatically.
  3. Make a basic text-completion request.
  4. Observe the trace that is captured.

Prerequisites
-------------
Install the required packages::

    pip install traceloop-sdk opentelemetry-instrumentation-alephalpha aleph_alpha_client python-dotenv

Set the environment variables (or create a .env file)::

    AA_TOKEN=<your-aleph-alpha-api-token>
    TRACELOOP_API_KEY=<your-traceloop-api-key>   # optional – omit to print traces locally

Expected trace
--------------
After running this script you should see a single span exported with:

  - span name           : "alephalpha.completion"
  - gen_ai.system       : "AlephAlpha"
  - llm.request.type    : "completion"
  - gen_ai.request.model: "luminous-base"
  - gen_ai.prompt.0.content  : <your prompt text>
  - gen_ai.completion.0.content : <the model's reply>
  - gen_ai.usage.input_tokens  : <number of prompt tokens>
  - gen_ai.usage.output_tokens : <number of generated tokens>
  - llm.usage.total_tokens     : input + output tokens
"""

import os

from aleph_alpha_client import Client, CompletionRequest, Prompt
from dotenv import load_dotenv
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

# ---------------------------------------------------------------------------
# Step 1 – Load environment variables from a .env file (if present)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Step 2 – Initialise Traceloop / OpenTelemetry
#
# Traceloop.init() sets up an OpenTelemetry TracerProvider and automatically
# instruments every supported LLM library that is installed, including the
# Aleph Alpha client.  No additional call to AlephAlphaInstrumentor is needed
# when using the SDK.
#
# If TRACELOOP_API_KEY is set the traces are sent to Traceloop's cloud.
# Otherwise they are printed to stdout (great for local development).
# ---------------------------------------------------------------------------
Traceloop.init(app_name="alephalpha_beginner_example")

# ---------------------------------------------------------------------------
# Step 3 – Create the Aleph Alpha client
# ---------------------------------------------------------------------------
aleph_alpha_client = Client(token=os.environ.get("AA_TOKEN", ""))


# ---------------------------------------------------------------------------
# Step 4 – Define a traced task that performs a single completion
#
# The @task decorator wraps the function in an OpenTelemetry span named
# "generate_joke".  The AlephAlpha instrumentation automatically adds a
# child span "alephalpha.completion" containing all LLM-specific attributes.
# ---------------------------------------------------------------------------
@task(name="generate_joke")
def generate_joke(prompt_text: str) -> str:
    """Send a prompt to Aleph Alpha and return the generated completion."""
    request = CompletionRequest(
        prompt=Prompt.from_text(prompt_text),
        maximum_tokens=200,
    )
    response = aleph_alpha_client.complete(request, model="luminous-base")
    completion = response.completions[0].completion
    print(f"\nCompletion received:\n{completion}\n")
    return completion


# ---------------------------------------------------------------------------
# Step 5 – Define a top-level workflow that calls the task
#
# The @workflow decorator creates the root span for this execution.  All
# child spans (tasks, LLM calls) are nested inside it, giving you a clear
# view of the full execution tree in your tracing backend.
# ---------------------------------------------------------------------------
@workflow(name="joke_generator")
def joke_generator():
    """Top-level workflow: ask the model for a joke and print it."""
    prompt = "Tell me a short, funny joke about observability."
    print(f"Prompt: {prompt}")
    joke = generate_joke(prompt)
    return joke


# ---------------------------------------------------------------------------
# Step 6 – Run the workflow
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Aleph Alpha LLM Tracing – Beginner Example")
    print("=" * 60)
    result = joke_generator()
    print("=" * 60)
    print("Done!  Check your Traceloop dashboard (or stdout) for the trace.")
    print("=" * 60)
