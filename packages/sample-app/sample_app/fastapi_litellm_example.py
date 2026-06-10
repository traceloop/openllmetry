import os

import litellm
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

load_dotenv()

Traceloop.init(app_name="fastapi_litellm_example", disable_batch=True)

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
    return response.choices[0].message.content

@workflow(name="chat_workflow")
def chat_workflow(message: str) -> str:
    return call_llm(message)

@app.post("/chat")
async def chat(request: ChatRequest):
    reply = chat_workflow(request.message)
    return {"reply": reply}

