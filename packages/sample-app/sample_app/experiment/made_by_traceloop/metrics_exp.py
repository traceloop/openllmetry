"""
Metrics Evaluators Experiment

This example demonstrates Traceloop's metrics evaluators:
- Character Count: Measure response length in characters
- Word Count: Measure response length in words
- Character Count Ratio: Compare lengths between texts
- Word Count Ratio: Compare word counts between texts

These evaluators help analyze and optimize response verbosity,
conciseness, and length consistency.
"""

import asyncio
import os
from openai import AsyncOpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop

# Initialize Traceloop
client = Traceloop.init()


async def generate_response(prompt: str, max_tokens: int = 200) -> str:
    """Generate a response using OpenAI"""
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


async def metrics_task(row):
    """
    Task function that generates responses and measures their metrics.
    """
    question = row.get("question", "")
    reference_answer = row.get("reference_answer", "This is a demo reference answer")

    # Generate response
    completion = await generate_response(question)

    # Return data for metrics evaluation
    return {
        "text": completion,
        "numerator_text": completion,
        "denominator_text": reference_answer,
    }


async def run_metrics_experiment():
    """
    Run experiment with metrics evaluators.

    This experiment measures:
    1. Character Count - Total characters in response
    2. Word Count - Total words in response
    3. Character Count Ratio - Response length vs reference
    4. Word Count Ratio - Response verbosity vs reference
    """

    print("\n" + "="*80)
    print("METRICS EVALUATORS EXPERIMENT")
    print("="*80 + "\n")

    print("This experiment measures response length and verbosity:\n")
    print("1. Character Count - Total characters in the response")
    print("2. Word Count - Total words in the response")
    print("3. Character Count Ratio - Response length compared to reference")
    print("4. Word Count Ratio - Response verbosity compared to reference")
    print("\n" + "-"*80 + "\n")

    # Configure metrics evaluators
    evaluators = [
        EvaluatorMadeByTraceloop.char_count(),
        EvaluatorMadeByTraceloop.word_count(),
        EvaluatorMadeByTraceloop.char_count_ratio(),
        EvaluatorMadeByTraceloop.word_count_ratio(),
    ]

    print("Running experiment with metrics evaluators:")
    for evaluator in evaluators:
        print(f"  - {evaluator.slug}")

    print("\n" + "-"*80 + "\n")

    # Run the experiment
    results, errors = await client.experiment.run(
        dataset_slug="metrics", # Set a ddataset slug that exists in the traceloop platform
        dataset_version="v1",
        task=metrics_task,
        evaluators=evaluators,
        experiment_slug="metrics-evaluators-exp",
        stop_on_error=False,
        wait_for_results=True,
    )

    print("\n" + "="*80)
    print("Metrics experiment completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\nMetrics Evaluators Experiment\n")

    asyncio.run(run_metrics_experiment())
