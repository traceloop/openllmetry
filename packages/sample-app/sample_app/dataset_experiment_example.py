import asyncio

from traceloop.sdk.datasets import Dataset


async def run_experiment_example(dataset_slug: str, dataset_version: str, evaluator_slug: str):
    """Demonstrate running an experiment"""
    print("\n=== Run Experiment Example ===")

    try:
        result = await Dataset.run_experiment(
            slug=dataset_slug,
            version=dataset_version,
            evaluator_slug=evaluator_slug,
            input_mapping={"love_only": "love_only", "love_sentence": "love_sentence"}, # {input_name: column_name}
        )

        print(f"Experiment result: {result}")

    except Exception as e:
        print(f"Error running experiment: {e}")


if __name__ == "__main__":
    asyncio.run(run_experiment_example(dataset_slug="ninja-1", dataset_version="v1", evaluator_slug="What I Hate"))