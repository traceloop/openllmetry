"""Async Bedrock + Traceloop — research-assistant flow.

A `@workflow` made of four `@task` steps. Each step calls a different async
Bedrock method through aioboto3, so the resulting trace has all four span
shapes nested under one workflow:

    research_assistant.workflow
    ├── classify_topic.task                       → converse (non-streaming)
    │   └── chat nova-lite-v1:0
    ├── generate_answer.task                      → converse_stream (streaming)
    │   └── chat nova-lite-v1:0
    ├── summarize_for_storage.task                → invoke_model (non-streaming)
    │   └── chat nova-lite-v1:0
    └── suggest_followups.task                    → invoke_model_with_response_stream (streaming)
        └── chat nova-lite-v1:0

Run this, then open the Traceloop dashboard — each task span shows the prompt,
the model output, tokens, and finish reason. The workflow span wraps them all
in a single trace so you can see the full flow end-to-end.
"""

import asyncio
import json

import aioboto3

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

Traceloop.init(app_name="async_bedrock_research_assistant")

MODEL_ID = "us.amazon.nova-lite-v1:0"


@task(name="classify_topic")
async def classify_topic(client, question: str) -> str:
    """Short, non-streaming call — fast `converse` to route the question."""
    response = await client.converse(
        modelId=MODEL_ID,
        system=[{"text": "Classify the question into exactly one word: 'tech', 'science', or 'history'."}],
        messages=[{"role": "user", "content": [{"text": question}]}],
        inferenceConfig={"maxTokens": 10, "temperature": 0.0},
    )
    return response["output"]["message"]["content"][0]["text"].strip().lower()


@task(name="generate_answer")
async def generate_answer(client, question: str, topic: str) -> str:
    """Streamed long-form answer — `converse_stream` for UI-friendly delivery."""
    response = await client.converse_stream(
        modelId=MODEL_ID,
        system=[{"text": f"You are an expert in {topic}. Answer in 2-3 sentences."}],
        messages=[{"role": "user", "content": [{"text": question}]}],
        inferenceConfig={"maxTokens": 300, "temperature": 0.5},
    )

    parts = []
    async for event in response["stream"]:
        if "contentBlockDelta" in event:
            text = event["contentBlockDelta"].get("delta", {}).get("text", "")
            if text:
                parts.append(text)
    return "".join(parts)


@task(name="summarize_for_storage")
async def summarize_for_storage(client, answer: str) -> str:
    """Structured single-shot — `invoke_model` to produce a short summary."""
    body = json.dumps({
        "schemaVersion": "messages-v1",
        "messages": [{
            "role": "user",
            "content": [{"text": f"Summarize this in under 15 words:\n\n{answer}"}],
        }],
        "inferenceConfig": {"maxTokens": 60, "temperature": 0.2},
    })
    response = await client.invoke_model(body=body, modelId=MODEL_ID)
    response_body = json.loads(await response["body"].read())
    return response_body["output"]["message"]["content"][0]["text"].strip()


@task(name="suggest_followups")
async def suggest_followups(client, question: str) -> list[str]:
    """Streamed structured generation — `invoke_model_with_response_stream`."""
    body = json.dumps({
        "schemaVersion": "messages-v1",
        "messages": [{
            "role": "user",
            "content": [{"text": f"Suggest 3 short follow-up questions to: '{question}'. One per line."}],
        }],
        "inferenceConfig": {"maxTokens": 200, "temperature": 0.7},
    })
    response = await client.invoke_model_with_response_stream(body=body, modelId=MODEL_ID)

    parts = []
    async for event in response["body"]:
        payload = json.loads(event["chunk"]["bytes"])
        if "contentBlockDelta" in payload:
            text = payload["contentBlockDelta"].get("delta", {}).get("text", "")
            if text:
                parts.append(text)
    return [line.strip("- ").strip() for line in "".join(parts).splitlines() if line.strip()]


@workflow(name="research_assistant")
async def research_assistant(question: str):
    session = aioboto3.Session()
    async with session.client("bedrock-runtime", region_name="us-east-1") as client:
        topic = await classify_topic(client, question)
        print(f"Topic:    {topic}")

        answer = await generate_answer(client, question, topic)
        print(f"Answer:   {answer}")

        summary = await summarize_for_storage(client, answer)
        print(f"Summary:  {summary}")

        followups = await suggest_followups(client, question)
        print("Followups:")
        for f in followups:
            print(f"  - {f}")


if __name__ == "__main__":
    asyncio.run(research_assistant("Why does the sky appear blue during the day?"))
