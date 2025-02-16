from fastapi import FastAPI
import httpx
from fastapi.responses import StreamingResponse
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task
from openai import AsyncOpenAI
import os

from traceloop.sdk.instruments import Instruments


# Initialize Traceloop
Traceloop.init(app_name="server1", disable_batch=False, block_instruments={Instruments.WATSONX})

app = FastAPI()
client = httpx.AsyncClient()
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@task(name="generate_response")
async def generate_response(prompt: str):
    stream = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


@app.get("/stream")
@workflow(name="test_openai_stream")
async def test_openai_stream():
    # Get references from server2
    # response = await client.get("http://localhost:8001/data")
    # references = response.json()
    references = [
        "Uvicorn is an ASGI web server implementation for Python.",
        "FastAPI is a modern, fast web framework for building APIs with Python 3.8+.",
        "Async functions in Python allow non-blocking I/O operations.",
        "ASGI servers like Uvicorn support WebSocket protocols.",
        "FastAPI uses Pydantic for data validation.",
        "Uvicorn's event loop enables high-performance async operations.",
        "Python's asyncio library is the foundation for async web servers.",
        "FastAPI automatically generates OpenAPI documentation."
    ]
    # Create the prompt with the references
    prompt = "Using these references about Python servers:\n\n"
    for ref in references:
        prompt += f"- {ref}\n"
    prompt += "\nExplain how to create a basic Python server using FastAPI and Uvicorn."

    response = generate_response(prompt)
    return StreamingResponse(response, media_type="text/plain")


@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()


def start():
    """Entry point for the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    start()
