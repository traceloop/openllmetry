"""
Example experiment script for CI/CD using run_in_github

This script:
1. Executes tasks locally on the dataset
2. Sends task results to the backend
3. Backend runs evaluators and posts PR comment with results
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
    print("🚀 Running research experiment in GitHub CI/CD...")

    # Execute tasks locally and send results to backend
    response = await client.experiment.run_in_github(
        task=research_task,
        dataset_slug="research-questions",
        dataset_version="v1",
        evaluators=["accuracy", "relevance"],
        experiment_slug="research-exp",
        stop_on_error=False,
    )

    # Print response
    print("\n✅ Experiment completed and submitted!")
    print(f"Experiment ID: {response.experiment_id}")
    print(f"Experiment Slug: {response.experiment_slug}")
    print(f"Run ID: {response.run_id}")
    print(f"Status: {response.status}")

    if response.message:
        print(f"Message: {response.message}")

    print("\n📝 The backend will run evaluators and post results to your PR.")
    print("   Check your GitHub PR for the results comment.")


if __name__ == "__main__":
    asyncio.run(main())
