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
        EvaluatorMadeByTraceloop.char_count(
            description="Count total characters in response"
        ),
        EvaluatorMadeByTraceloop.word_count(
            description="Count total words in response"
        ),
        EvaluatorMadeByTraceloop.char_count_ratio(
            description="Compare response length to reference"
        ),
        EvaluatorMadeByTraceloop.word_count_ratio(
            description="Compare response verbosity to reference"
        ),
    ]

    print("Running experiment with metrics evaluators:")
    for evaluator in evaluators:
        print(f"  - {evaluator.slug}")

    print("\n" + "-"*80 + "\n")

    # Run the experiment
    results, errors = await client.experiment.run(
        dataset_slug="medical-q",
        dataset_version="v1",
        task=metrics_task,
        evaluators=evaluators,
        experiment_slug="metrics-evaluators-exp",
        stop_on_error=False,
        wait_for_results=True,
    )

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")

    if results:
        print(f"Successfully evaluated {len(results)} tasks\n")

        # Collect metrics
        all_metrics = {
            "char_counts": [],
            "word_counts": [],
            "char_ratios": [],
            "word_ratios": [],
        }

        for i, result in enumerate(results, 1):
            print(f"Task {i}:")
            if result.task_result:
                text = result.task_result.get("text", "")
                print(f"  Response: {text[:60]}{'...' if len(text) > 60 else ''}")

            if result.evaluations:
                for eval_name, eval_result in result.evaluations.items():
                    print(f"  {eval_name}: {eval_result}")

                    # Collect metrics for statistics
                    if "char-count" in eval_name and isinstance(eval_result, dict):
                        if "char_count" in eval_result:
                            all_metrics["char_counts"].append(eval_result["char_count"])
                        elif "char_ratio" in eval_result:
                            all_metrics["char_ratios"].append(eval_result["char_ratio"])
                    elif "word-count" in eval_name and isinstance(eval_result, dict):
                        if "word_count" in eval_result:
                            all_metrics["word_counts"].append(eval_result["word_count"])
                        elif "word_ratio" in eval_result:
                            all_metrics["word_ratios"].append(eval_result["word_ratio"])
            print()

        # Metrics Summary
        print("\n" + "="*80)
        print("METRICS SUMMARY")
        print("="*80 + "\n")

        if all_metrics["char_counts"]:
            avg_chars = sum(all_metrics["char_counts"]) / len(all_metrics["char_counts"])
            min_chars = min(all_metrics["char_counts"])
            max_chars = max(all_metrics["char_counts"])
            print(f"Character Count Statistics:")
            print(f"  Average: {avg_chars:.0f} chars")
            print(f"  Range: {min_chars} - {max_chars} chars\n")

        if all_metrics["word_counts"]:
            avg_words = sum(all_metrics["word_counts"]) / len(all_metrics["word_counts"])
            min_words = min(all_metrics["word_counts"])
            max_words = max(all_metrics["word_counts"])
            print(f"Word Count Statistics:")
            print(f"  Average: {avg_words:.0f} words")
            print(f"  Range: {min_words} - {max_words} words\n")

        if all_metrics["char_ratios"]:
            avg_char_ratio = sum(all_metrics["char_ratios"]) / len(all_metrics["char_ratios"])
            print(f"Character Ratio (Response/Reference):")
            print(f"  Average: {avg_char_ratio:.2f}x")
            if avg_char_ratio > 1.5:
                print(f"  Note: Responses are {avg_char_ratio:.1f}x longer than reference\n")
            elif avg_char_ratio < 0.5:
                print(f"  Note: Responses are {avg_char_ratio:.1f}x shorter than reference\n")
            else:
                print(f"  Note: Response length is well-matched to reference\n")

        if all_metrics["word_ratios"]:
            avg_word_ratio = sum(all_metrics["word_ratios"]) / len(all_metrics["word_ratios"])
            print(f"Word Ratio (Response/Reference):")
            print(f"  Average: {avg_word_ratio:.2f}x")

        print("\nInsights:")
        if all_metrics["word_counts"]:
            avg_words = sum(all_metrics["word_counts"]) / len(all_metrics["word_counts"])
            if avg_words < 50:
                print("  - Responses are concise")
            elif avg_words > 150:
                print("  - Responses are verbose")
            else:
                print("  - Response length is moderate")

    else:
        print("No results to display")

    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for error in errors[:5]:
            print(f"  - {error}")
    else:
        print("\nNo errors encountered")

    print("\n" + "="*80)
    print("Metrics experiment completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\nMetrics Evaluators Experiment\n")

    asyncio.run(run_metrics_experiment())
