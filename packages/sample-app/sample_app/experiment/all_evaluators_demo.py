"""
Comprehensive demo of all Traceloop predefined evaluators.

This example demonstrates all the predefined evaluators available in Traceloop,
showing how to configure and use each one.
"""

import asyncio
from traceloop.sdk import Traceloop
from traceloop.sdk.evaluator import Predefined

client = Traceloop.init()


async def sample_task(row):
    """Sample task that generates text output for evaluation"""
    return {
        "text": "Hello, my name is John Smith and my email is john.smith@example.com. "
                "This is a test response with some SQL: SELECT * FROM users WHERE id = 1",
        "question": row.get("question", "What is your name?"),
        "answer": "My name is John Smith.",
        "completion": "The capital of France is Paris.",
        "context": "France is a country in Europe. Paris is its capital city.",
        "reference": "Paris is the capital of France.",
        "numerator_text": "This is a longer text with more words and characters.",
        "denominator_text": "Short text.",
        "reference_topics": "geography, capitals, France",
        "logprobs": "[-0.5, -0.3, -0.2, -0.1]",
        "placeholder_value": "test-value-123",
    }


async def run_all_evaluators_demo():
    """Demonstrate all Traceloop predefined evaluators"""

    print("\n" + "="*80)
    print("TRACELOOP PREDEFINED EVALUATORS COMPREHENSIVE DEMO")
    print("="*80 + "\n")

    # Create a list of all evaluators with their configurations
    evaluators = [
        # === Metrics Evaluators ===
        Predefined.char_count(),
        Predefined.char_count_ratio(),
        Predefined.word_count(),
        Predefined.word_count_ratio(),

        # === Quality Evaluators ===
        Predefined.answer_relevancy(),
        Predefined.faithfulness(),
        Predefined.semantic_similarity(),

        # === Security Evaluators ===
        Predefined.pii_detector(probability_threshold=0.7),
        Predefined.profanity_detector(),
        Predefined.secrets_detector(),
        Predefined.prompt_injection(threshold=0.6),

        # === Validation Evaluators ===
        Predefined.json_validator(enable_schema_validation=False),
        Predefined.sql_validator(),
        Predefined.regex_validator(
            regex=r".*@.*\.com",
            should_match=True,
            case_sensitive=False,
        ),
        Predefined.placeholder_regex(
            regex=r"test-.*",
            placeholder_name="test_placeholder",
            should_match=True,
        ),

        # === Agent & Topic Evaluators ===
        Predefined.agent_goal_accuracy(),
        Predefined.topic_adherence(),

        # === Advanced Evaluators ===
        Predefined.perplexity(),

        # === LLM as Judge (commented out - requires API keys) ===
        # Predefined.llm_as_judge(
        #     messages=[
        #         {"role": "system", "content": "You are an evaluator."},
        #         {"role": "user", "content": "Evaluate this: {{text}}"}
        #     ],
        #     provider="openai",
        #     model="gpt-4",
        #     temperature=0.3,
        # ),
    ]

    print(f"🔬 Running experiment with {len(evaluators)} evaluators...\n")
    print("Evaluators being tested:")
    for i, evaluator in enumerate(evaluators, 1):
        print(f"  {i}. {evaluator.slug}")

    print("\n" + "-"*80 + "\n")

    # Run the experiment with all evaluators
    results, errors = await client.experiment.run(
        dataset_slug="demo-dataset",  # You'll need to create this dataset
        dataset_version="v1",
        task=sample_task,
        evaluators=evaluators,
        experiment_slug="all-evaluators-demo",
        stop_on_error=False,
        wait_for_results=True,
    )

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")

    if results:
        print(f"✅ Successfully evaluated {len(results)} tasks\n")
        for i, result in enumerate(results[:3], 1):  # Show first 3 results
            print(f"Task {i}:")
            if result.evaluations:
                for eval_name, eval_result in result.evaluations.items():
                    print(f"  - {eval_name}: {eval_result}")
            print()
    else:
        print("ℹ️  No results to display (possibly running in fire-and-forget mode)\n")

    if errors:
        print(f"⚠️  Encountered {len(errors)} errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
    else:
        print("✅ No errors encountered")

    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80 + "\n")


async def run_categorized_evaluators_demo():
    """
    Alternative demo showing evaluators organized by category.
    This helps understand when to use each type of evaluator.
    """

    print("\n" + "="*80)
    print("CATEGORIZED EVALUATORS DEMO")
    print("="*80 + "\n")

    categories = {
        "📊 Metrics & Counting": [
            Predefined.char_count(),
            Predefined.word_count(),
        ],

        "🔒 Security & Safety": [
            Predefined.pii_detector(probability_threshold=0.8),
            Predefined.secrets_detector(),
            Predefined.prompt_injection(threshold=0.7),
        ],

        "✅ Content Validation": [
            Predefined.json_validator(),
            Predefined.sql_validator(),
            Predefined.regex_validator(regex=r"^\d{3}-\d{2}-\d{4}$", should_match=False),
        ],

        "🎯 Quality & Relevance": [
            Predefined.answer_relevancy(),
            Predefined.faithfulness(),
            Predefined.semantic_similarity(),
        ],

        "🤖 Agent & Topics": [
            Predefined.agent_goal_accuracy(),
            Predefined.topic_adherence(),
        ],
    }

    for category, evaluators in categories.items():
        print(f"\n{category}:")
        for evaluator in evaluators:
            print(f"  • {evaluator.slug}")
            if evaluator.config:
                config_str = ", ".join(f"{k}={v}" for k, v in evaluator.config.items() if k != "description")
                if config_str:
                    print(f"    Config: {config_str}")

    print("\n" + "="*80)
    print("\nYou can use these evaluators in your experiments like this:")
    print("""
    results, errors = await client.experiment.run(
        dataset_slug="your-dataset",
        dataset_version="v1",
        task=your_task_function,
        evaluators=[
            Predefined.pii_detector(probability_threshold=0.8),
            Predefined.answer_relevancy(),
            "your-custom-evaluator",  # Mix with custom evaluators
        ],
        experiment_slug="my-experiment",
    )
    """)
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n🚀 Starting Traceloop Evaluators Demo\n")
    print("Choose a demo to run:")
    print("1. Comprehensive demo (all evaluators)")
    print("2. Categorized demo (organized by use case)")

    # For now, run the categorized demo which doesn't require dataset
    asyncio.run(run_categorized_evaluators_demo())

    # To run the comprehensive demo, uncomment:
    # asyncio.run(run_all_evaluators_demo())
