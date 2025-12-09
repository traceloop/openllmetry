"""
Quality Evaluators Experiment

This example demonstrates Traceloop's correctness evaluators:
- Answer Relevancy: Verifies responses address the query
- Faithfulness: Detects hallucinations and verifies facts

These evaluators help ensure your AI applications provide accurate,
relevant, and faithful responses.
"""

import asyncio
import os
from openai import AsyncOpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop

# Initialize Traceloop
client = Traceloop.init()


async def generate_response(prompt: str, context: str = None) -> str:
    """Generate a response using OpenAI"""
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [{"role": "user", "content": prompt}]
    if context:
        messages.insert(0, {"role": "system", "content": f"Context: {context}"})

    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=200,
    )

    return response.choices[0].message.content


async def quality_task(row):
    """
    Task function that processes questions with context.
    Returns data that will be evaluated for quality and faithfulness.
    """
    question = row.get("question", "This is a demo question")
    context = row.get("context", "This is a demo context")
    # Generate response
    completion = await generate_response(question, context)

    # Return data for evaluation
    return {
        "question": question,
        "answer": completion,
        "completion": completion,
        "context": context,
    }


async def run_correctness_experiment():
    """
    Run experiment with correctness evaluators.

    This experiment will evaluate responses for:
    1. Answer Relevancy - Does the answer address the question?
    2. Faithfulness - Is the answer faithful to the provided context?
    """

    print("\n" + "="*80)
    print("CORRECTNESS EVALUATORS EXPERIMENT")
    print("="*80 + "\n")

    print("This experiment will test two critical quality evaluators:\n")
    print("1. Answer Relevancy - Verifies the response addresses the query")
    print("2. Faithfulness - Detects hallucinations and verifies factual accuracy")
    print("\n" + "-"*80 + "\n")

    # Configure correctness evaluators
    evaluators = [
        EvaluatorMadeByTraceloop.answer_relevancy(),
        EvaluatorMadeByTraceloop.faithfulness(),
    ]

    print("Running experiment with evaluators:")
    for evaluator in evaluators:
        print(f"  - {evaluator.slug}")

    print("\n" + "-"*80 + "\n")

    # Run the experiment
    results, errors = await client.experiment.run(
        dataset_slug="correctness",  # Set a dataset slug that exists in the traceloop platform
        dataset_version="v1",
        task=quality_task,
        evaluators=evaluators,
        experiment_slug="correctness-evaluators-exp",
        stop_on_error=False,
        wait_for_results=True,
    )

    print("\n" + "="*80)
    print("Correctness experiment completed!")
    print("="*80 + "\n")

if __name__ == "__main__":
    print("\nCorrectness Evaluators Experiment\n")

    asyncio.run(run_correctness_experiment())
