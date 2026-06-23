import os

import litellm
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from pydantic import BaseModel

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

load_dotenv()

Traceloop.init(app_name="fastapi_litellm_example", disable_batch=True, exporter=ConsoleSpanExporter())

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@task(name="call_llm")
def call_llm(message: str) -> str:
    response = litellm.completion(
        model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
        messages=[{"role": "user", "content": message}],
        api_base=os.environ.get("LLM_API_BASE", None),
    )
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("Empty response from LLM")
    return response.choices[0].message.content

@workflow(name="chat_workflow")
def chat_workflow(message: str) -> str:
    return call_llm(message)

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        reply = chat_workflow(request.message)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"reply": reply}

