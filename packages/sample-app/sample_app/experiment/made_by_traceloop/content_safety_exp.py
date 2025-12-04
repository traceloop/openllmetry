"""
Content Safety Evaluators Experiment

This example demonstrates Traceloop's content safety evaluators:
- Profanity Detection: Flags inappropriate language
- Toxicity Detection: Identifies toxic or harmful content
- (Note: Sexism Detection is planned but not yet implemented)

These evaluators help ensure AI-generated content is safe,
appropriate, and adheres to community guidelines.
"""

import asyncio
import os
from openai import AsyncOpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop

# Initialize Traceloop
client = Traceloop.init()


async def generate_response(prompt: str, temperature: float = 0.7) -> str:
    """Generate a response using OpenAI"""
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=200,
    )

    return response.choices[0].message.content


async def content_safety_task(row):
    """
    Task function that generates content to be evaluated for safety.
    Returns text that will be checked for profanity and toxicity.
    """
    prompt = row.get("question", "")

    # Generate response
    response = await generate_response(prompt)

    # Return data for safety evaluation
    return {
        "text": response,
        "prompt": prompt,
    }


async def run_content_safety_experiment():
    """
    Run experiment with content safety evaluators.

    This experiment evaluates:
    1. Profanity Detection - Flags inappropriate language
    2. Toxicity Detection - Identifies harmful or toxic content
    """

    print("\n" + "="*80)
    print("CONTENT SAFETY EVALUATORS EXPERIMENT")
    print("="*80 + "\n")

    print("This experiment tests content safety:\n")
    print("1. Profanity Detection - Identifies inappropriate language")
    print("2. Toxicity Detection - Detects harmful, aggressive, or toxic content")
    print("\n" + "-"*80 + "\n")

    # Configure content safety evaluators
    evaluators = [
        EvaluatorMadeByTraceloop.profanity_detector(),
        EvaluatorMadeByTraceloop.toxicity_detector(threshold=0.7),
        EvaluatorMadeByTraceloop.sexism_detector(threshold=0.7),
    ]

    print("Running experiment with content safety evaluators:")
    for evaluator in evaluators:
        config_str = ", ".join(f"{k}={v}" for k, v in evaluator.config.items() if k != "description")
        print(f"  - {evaluator.slug}")
        if config_str:
            print(f"    Config: {config_str}")

    print("\n" + "-"*80 + "\n")

    # Run the experiment
    results, errors = await client.experiment.run(
        dataset_slug="medical-q",
        dataset_version="v1",
        task=content_safety_task,
        evaluators=evaluators,
        experiment_slug="content-safety-exp",
        stop_on_error=False,
        wait_for_results=True,
    )

    print("\n" + "="*80)
    print("Content safety experiment completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\nContent Safety Evaluators Experiment\n")


    asyncio.run(run_content_safety_experiment())
