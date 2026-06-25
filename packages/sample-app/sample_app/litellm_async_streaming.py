"""Async + streaming LiteLLM completion instrumented by Traceloop.

This mirrors the primary call shape used by agent-orchestrator-v2:
`await litellm.acompletion(..., stream=True)`. The instrumentation accumulates the
streamed deltas into a single `litellm.chat` span (content + token usage) once the
stream is fully consumed.

Pass `stream_options={"include_usage": True}` so the provider reports token usage on
the final chunk.

Requires: OPENAI_API_KEY
"""

import asyncio

import litellm
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow
from traceloop.sdk.instruments import Instruments

Traceloop.init(
    app_name="litellm_async_streaming_example",
    instruments={Instruments.LITELLM},
)


@task(name="stream_answer")
async def stream_answer(question: str) -> str:
    stream = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}],
        max_tokens=150,
        temperature=0.7,
        stream=True,
        stream_options={"include_usage": True},
    )

    chunks = []
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        chunks.append(delta)
        print(delta, end="", flush=True)
    print()
    return "".join(chunks)


@workflow(name="litellm_async_streaming")
async def main():
    await stream_answer("Explain distributed tracing to a five year old.")


if __name__ == "__main__":
    asyncio.run(main())
