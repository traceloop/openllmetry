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
from traceloop.sdk.experiment.model import RunInGithubResponse

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
    query = row.get("query", "")
    answer = await generate_research_response(query)

    return {
        "completion": answer,
        "question": query,
        "text": answer
    }


async def main():
    """Run experiment in GitHub context"""
    print("🚀 Running research experiment in GitHub CI/CD...")

    # Execute tasks locally and send results to backend
    response = await client.experiment.run(
        task=research_task,
        dataset_slug="research-queries",
        dataset_version="v2",
        evaluators=["research-relevancy", "categories", "research-facts-counter"],
        experiment_slug="research-exp",
    )

    # Print response
    print("\n✅ Experiment completed and submitted!")

    if isinstance(response, RunInGithubResponse):
        print(f"Experiment Slug: {response.experiment_slug}")
        print(f"Run ID: {response.run_id}")
    else:
        print(f"Results: {response}")

    print("\n📝 The backend will run evaluators and post results to your PR.")
    print("   Check your GitHub PR for the results comment.")


if __name__ == "__main__":
    asyncio.run(main())
