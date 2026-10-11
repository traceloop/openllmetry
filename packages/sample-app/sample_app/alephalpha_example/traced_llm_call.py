"""
Beginner-friendly LLM tracing example for OpenLLMetry.

This example shows how to trace a basic LLM call step-by-step using
OpenLLMetry (Traceloop SDK). No real API key or network call needed —
we simulate the LLM response so you can focus on the tracing pattern.

What you'll learn:
  1. What LLM tracing / observability is
  2. How to initialize Traceloop with a console exporter
  3. How to manually trace an LLM call using track_llm_call
  4. How spans capture request, response, and usage metadata

The console exporter prints span JSON to stdout, so you can see exactly
what an observability backend (Jaeger, Grafana, Datadog, etc.) would
receive — without signing up for anything.
"""

from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow
from traceloop.sdk.tracing.manual import LLMMessage, LLMUsage, track_llm_call


def simulate_llm_call(model: str, messages: list[dict]) -> dict:
    """Simulate an LLM response.

    In a real application this would be an actual provider call, e.g.:
        client = openai.OpenAI()
        response = client.chat.completions.create(...)

    We mock it here so the example runs without any API key.
    """
    user_message = messages[0]["content"] if messages else ""
    mock_response = (
        f"This is a simulated response to: '{user_message}' "
        f"(model={model})"
    )

    # Simulated token counts (in a real response these come from the provider)
    prompt_tokens = len(user_message.split()) if user_message else 5
    completion_tokens = len(mock_response.split())
    total_tokens = prompt_tokens + completion_tokens

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": mock_response,
                }
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


@task(name="generate_response")
def generate_response(question: str) -> dict:
    """Task: send a question to an LLM and return the response.

    The @task decorator automatically wraps this function in a span
    named 'generate_response'. You'll see it in the trace output.
    """
    messages = [{"role": "user", "content": question}]
    response = simulate_llm_call(model="gpt-4o", messages=messages)
    return response


@workflow(name="beginner_tracing_example")
def beginner_tracing_example():
    """Workflow: orchestrate the end-to-end tracing demonstration.

    The @workflow decorator creates a parent span that links together
    all the tasks and LLM calls inside it. This gives you a tree view
    of your application's execution in your observability backend.
    """

    # ------------------------------------------------------------------
    # Example 1: Basic LLM call with manual tracing
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Example 1: Basic LLM call with manual tracing")
    print("=" * 60)

    # track_llm_call is a context manager that creates a span for a
    # single LLM interaction. You report request/response/usage inside
    # the `with` block and OpenLLMetry handles the rest.
    with track_llm_call(vendor="openai", type="chat") as span:
        # Step 1: Report the request (what you send to the LLM)
        messages = [LLMMessage(role="user", content="What is observability?")]
        span.report_request(model="gpt-4o", messages=messages)

        # Step 2: Make the actual LLM call (simulated here)
        response = simulate_llm_call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What is observability?"}],
        )

        # Step 3: Report the response (what the LLM returned)
        completions = [
            choice["message"]["content"]
            for choice in response["choices"]
        ]
        span.report_response(model="gpt-4o", completions=completions)

        # Step 4: Report usage (token counts for cost/perf tracking)
        usage = response["usage"]
        span.report_usage(
            LLMUsage(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
            )
        )

        print(f"Response: {completions[0]}")

    # ------------------------------------------------------------------
    # Example 2: Using decorators (@task + @workflow)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Example 2: Using @task and @workflow decorators")
    print("=" * 60)

    question = "Explain OpenTelemetry in one sentence."
    result = generate_response(question)
    answer = result["choices"][0]["message"]["content"]
    print(f"Q: {question}")
    print(f"A: {answer}")

    # ------------------------------------------------------------------
    # What you should see in the console output
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("What just happened (the trace):")
    print("=" * 60)
    print(
        "Look above this message — you should see JSON blobs printed by\n"
        "ConsoleSpanExporter. Each blob is a 'span' representing one\n"
        "operation. Key fields to notice:"
    )
    print("  - name: the span name (e.g. 'openai.chat', 'generate_response')")
    print("  - attributes: gen_ai.request.model, gen_ai.usage.*, etc.")
    print("  - trace_id / span_id: links spans into a tree")
    print()
    print(
        "In production, these spans go to Jaeger, Grafana Tempo, Datadog,\n"
        "or the Traceloop platform — not the console."
    )


def main():
    """Run the beginner LLM tracing example."""
    # Initialize Traceloop with console export so you can see traces
    # immediately, without any API key or external service.
    Traceloop.init(
        app_name="beginner-tracing-example",
        exporter=ConsoleSpanExporter(),
        disable_batch=True,  # flush immediately for demo clarity
    )

    beginner_tracing_example()


if __name__ == "__main__":
    main()
