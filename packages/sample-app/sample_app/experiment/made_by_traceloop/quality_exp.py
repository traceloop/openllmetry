"""
Advanced Quality Evaluators Experiment

This example demonstrates Traceloop's advanced quality evaluators:
- Measure Perplexity: Measure text perplexity from logprobs
- Agent Goal Accuracy: Validate agent goal achievement
- Semantic Similarity: Measure semantic similarity between texts
- Topic Adherence: Validate topic adherence

These evaluators help analyze response quality, goal achievement,
semantic correctness, and topic consistency.
"""

import asyncio
import os
from openai import AsyncOpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop

# Initialize Traceloop
client = Traceloop.init()


async def generate_response(prompt: str, max_tokens: int = 300) -> str:
    """Generate a response using OpenAI with logprobs for perplexity measurement"""
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_tokens,
        logprobs=True,  # Enable logprobs for perplexity calculation
    )

    return response.choices[0].message.content


async def advanced_quality_task(row):
    """
    Task function that generates responses and provides data for advanced quality evaluation.

    Expected dataset fields:
    - question: The input question/prompt
    - expected_goal: The expected outcome or goal (for agent goal accuracy)
    - reference_answer: Reference answer for semantic similarity
    - topic: The expected topic for adherence checking
    """
    question = row.get("question", "")
    reference_answer = row.get("reference_answer", "This is a demo reference answer")
    topic = row.get("topic", "general knowledge")

    # Generate response
    completion = await generate_response(question)

    # Return data for evaluation
    # Different evaluators expect different fields
    return {
        "logprobs": completion,  # For perplexity
        "question": question,
        "completion": completion,  # Standard completion field
        "reference": reference_answer,  # For semantic similarity comparison
        "reference_topics": topic,  # For topic adherence
    }


async def run_advanced_quality_experiment():
    """
    Run experiment with advanced quality evaluators.

    This experiment measures:
    1. Perplexity - Text fluency and predictability from logprobs
    2. Agent Goal Accuracy - Whether the response achieves its intended goal
    3. Semantic Similarity - Semantic alignment with reference answer
    4. Topic Adherence - Whether the response stays on topic
    """

    print("\n" + "="*80)
    print("ADVANCED QUALITY EVALUATORS EXPERIMENT")
    print("="*80 + "\n")

    print("This experiment measures response quality across multiple dimensions:\n")
    print("1. Perplexity - Measures text fluency and predictability")
    print("2. Agent Goal Accuracy - Validates goal achievement")
    print("3. Semantic Similarity - Compares semantic meaning with reference")
    print("4. Topic Adherence - Ensures response stays on topic")
    print("\n" + "-"*80 + "\n")

    # Configure advanced quality evaluators
    evaluators = [
        EvaluatorMadeByTraceloop.perplexity(),
        EvaluatorMadeByTraceloop.agent_goal_accuracy(),
        EvaluatorMadeByTraceloop.semantic_similarity(),
        EvaluatorMadeByTraceloop.topic_adherence(),
    ]

    print("Running experiment with advanced quality evaluators:")
    for evaluator in evaluators:
        print(f"  - {evaluator.slug}")

    print("\n" + "-"*80 + "\n")

    # Run the experiment
    results, errors = await client.experiment.run(
        dataset_slug="quality",  # Set a dataset slug that exists in the traceloop platform
        dataset_version="v1",
        task=advanced_quality_task,
        evaluators=evaluators,
        experiment_slug="advanced-quality-exp",
        stop_on_error=False,
        wait_for_results=True,
    )

    print("\n" + "="*80)
    print("Advanced quality experiment completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n🚀 Advanced Quality Evaluators Experiment\n")

    asyncio.run(run_advanced_quality_experiment())
