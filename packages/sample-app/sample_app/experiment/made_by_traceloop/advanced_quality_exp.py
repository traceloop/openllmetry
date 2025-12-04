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
        dataset_slug="medical-q",
        dataset_version="v1",
        task=advanced_quality_task,
        evaluators=evaluators,
        experiment_slug="advanced-quality-exp",
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
            "perplexities": [],
            "goal_accuracies": [],
            "semantic_similarities": [],
            "topic_adherences": [],
        }

        for i, result in enumerate(results, 1):
            print(f"Task {i}:")
            if result.task_result:
                text = result.task_result.get("text", "")
                question = result.task_result.get("question", "")
                print(f"  Question: {question[:60]}{'...' if len(question) > 60 else ''}")
                print(f"  Response: {text[:60]}{'...' if len(text) > 60 else ''}")

            if result.evaluations:
                for eval_name, eval_result in result.evaluations.items():
                    print(f"  {eval_name}: {eval_result}")

                    # Collect metrics for statistics
                    if "perplexity" in eval_name and isinstance(eval_result, dict):
                        if "perplexity" in eval_result:
                            all_metrics["perplexities"].append(eval_result["perplexity"])
                    elif "agent-goal-accuracy" in eval_name and isinstance(eval_result, dict):
                        if "score" in eval_result:
                            all_metrics["goal_accuracies"].append(eval_result["score"])
                    elif "semantic-similarity" in eval_name and isinstance(eval_result, dict):
                        if "score" in eval_result:
                            all_metrics["semantic_similarities"].append(eval_result["score"])
                    elif "topic-adherence" in eval_name and isinstance(eval_result, dict):
                        if "score" in eval_result:
                            all_metrics["topic_adherences"].append(eval_result["score"])
            print()

        # Advanced Quality Summary
        print("\n" + "="*80)
        print("QUALITY METRICS SUMMARY")
        print("="*80 + "\n")

        if all_metrics["perplexities"]:
            avg_perplexity = sum(all_metrics["perplexities"]) / len(all_metrics["perplexities"])
            min_perplexity = min(all_metrics["perplexities"])
            max_perplexity = max(all_metrics["perplexities"])
            print(f"Perplexity Statistics:")
            print(f"  Average: {avg_perplexity:.2f}")
            print(f"  Range: {min_perplexity:.2f} - {max_perplexity:.2f}")
            if avg_perplexity < 10:
                print(f"  Assessment: Excellent fluency and predictability")
            elif avg_perplexity < 50:
                print(f"  Assessment: Good fluency")
            else:
                print(f"  Assessment: Higher perplexity - may indicate complexity or uncertainty")
            print()

        if all_metrics["goal_accuracies"]:
            avg_goal_accuracy = sum(all_metrics["goal_accuracies"]) / len(all_metrics["goal_accuracies"])
            min_goal = min(all_metrics["goal_accuracies"])
            max_goal = max(all_metrics["goal_accuracies"])
            print(f"Agent Goal Accuracy Statistics:")
            print(f"  Average: {avg_goal_accuracy:.2%}")
            print(f"  Range: {min_goal:.2%} - {max_goal:.2%}")
            if avg_goal_accuracy >= 0.8:
                print(f"  Assessment: Strong goal achievement")
            elif avg_goal_accuracy >= 0.6:
                print(f"  Assessment: Moderate goal achievement")
            else:
                print(f"  Assessment: Needs improvement in goal achievement")
            print()

        if all_metrics["semantic_similarities"]:
            avg_similarity = sum(all_metrics["semantic_similarities"]) / len(all_metrics["semantic_similarities"])
            min_sim = min(all_metrics["semantic_similarities"])
            max_sim = max(all_metrics["semantic_similarities"])
            print(f"Semantic Similarity Statistics:")
            print(f"  Average: {avg_similarity:.2%}")
            print(f"  Range: {min_sim:.2%} - {max_sim:.2%}")
            if avg_similarity >= 0.8:
                print(f"  Assessment: High semantic alignment with reference")
            elif avg_similarity >= 0.6:
                print(f"  Assessment: Moderate semantic alignment")
            else:
                print(f"  Assessment: Low semantic alignment - responses differ significantly")
            print()

        if all_metrics["topic_adherences"]:
            avg_adherence = sum(all_metrics["topic_adherences"]) / len(all_metrics["topic_adherences"])
            min_adh = min(all_metrics["topic_adherences"])
            max_adh = max(all_metrics["topic_adherences"])
            print(f"Topic Adherence Statistics:")
            print(f"  Average: {avg_adherence:.2%}")
            print(f"  Range: {min_adh:.2%} - {max_adh:.2%}")
            if avg_adherence >= 0.8:
                print(f"  Assessment: Excellent topic consistency")
            elif avg_adherence >= 0.6:
                print(f"  Assessment: Moderate topic focus")
            else:
                print(f"  Assessment: Responses frequently drift off-topic")
            print()

        print("Overall Insights:")
        insights = []

        if all_metrics["perplexities"]:
            avg_perplexity = sum(all_metrics["perplexities"]) / len(all_metrics["perplexities"])
            if avg_perplexity < 20:
                insights.append("  - Responses show strong fluency and confidence")

        if all_metrics["goal_accuracies"]:
            avg_goal = sum(all_metrics["goal_accuracies"]) / len(all_metrics["goal_accuracies"])
            if avg_goal >= 0.7:
                insights.append("  - Responses effectively achieve intended goals")
            else:
                insights.append("  - Consider refining prompts to better achieve goals")

        if all_metrics["semantic_similarities"]:
            avg_sim = sum(all_metrics["semantic_similarities"]) / len(all_metrics["semantic_similarities"])
            if avg_sim >= 0.7:
                insights.append("  - Responses align well with reference answers")

        if all_metrics["topic_adherences"]:
            avg_adh = sum(all_metrics["topic_adherences"]) / len(all_metrics["topic_adherences"])
            if avg_adh >= 0.7:
                insights.append("  - Responses maintain good topic focus")
            else:
                insights.append("  - Consider adding topic constraints to prompts")

        if insights:
            for insight in insights:
                print(insight)
        else:
            print("  - Collect more data for comprehensive insights")

    else:
        print("No results to display")

    if errors:
        print(f"\n⚠️  Encountered {len(errors)} errors:")
        for error in errors[:5]:
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    else:
        print("\n✅ No errors encountered")

    print("\n" + "="*80)
    print("Advanced quality experiment completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n🚀 Advanced Quality Evaluators Experiment\n")

    asyncio.run(run_advanced_quality_experiment())
