from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from traceloop.sdk import Traceloop

Traceloop.init(app_name="fastapi_test", disable_batch=True)
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class Message(BaseModel):
    content: str


class LLM:
    def __init__(self):
        # self.callback = AsyncIteratorCallbackHandler()
        self.model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=100,
            streaming=True,
            # callbacks=[self.callback],
        )

    async def run_llm(self, question: str):
        yield "Answer:\n"
        async for chunk in self.model.astream(
            "hello. tell me something about yourself"
        ):
            yield chunk.content


@app.post("/stream_chat")
def stream_chat(message: Message):
    llm = LLM()
    llm_generator = llm.run_llm(message.content)
    return StreamingResponse(llm_generator, media_type="text/event-stream")
