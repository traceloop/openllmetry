"""
Example experiment script for CI/CD using run_in_github
"""

import asyncio
import os
from openai import AsyncOpenAI
from traceloop.sdk import Traceloop

# Initialize Traceloop client
client = Traceloop.init(
    app_name="research-experiment-ci-cd",
    api_key=os.getenv("TRACELOOP_API_KEY"),
    api_endpoint=os.getenv("TRACELOOP_BASE_URL"),
)


async def generate_research_response(question: str) -> str:
    """Generate a research response using OpenAI"""
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful research assistant. Provide accurate, well-researched answers.",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_tokens=500,
    )

    return response.choices[0].message.content


async def research_task(row):
    """Task function that processes each dataset row"""
    question = row.get("question", "")
    answer = await generate_research_response(question)

    print(f"Question: {question}")
    print(f"Answer: {answer[:100]}...")

    return {
        "completion": answer,
        "question": question,
    }


async def main():
    """Run experiment in GitHub context"""
    print("🚀 Starting research experiment in GitHub CI/CD...")

    # Run experiment using run_in_github which automatically captures GitHub context
    results, errors = await client.experiment.run_in_github(
        dataset_slug="research-questions",
        dataset_version="v1",
        task=research_task,
        evaluators=["accuracy", "relevance"],
        experiment_slug="research-exp",
        stop_on_error=False,
        wait_for_results=True,
    )

    # Print results
    print(f"\n✅ Experiment completed!")
    print(f"Total results: {len(results)}")

    if results:
        print(f"\nSample result:")
        print(f"  Task output: {results[0].task_result}")
        print(f"  Evaluations: {results[0].evaluations}")

    if errors:
        print(f"\n⚠️ Errors encountered: {len(errors)}")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        exit(1)

    print("\n🎉 All tasks completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
