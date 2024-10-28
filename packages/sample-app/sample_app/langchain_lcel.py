import asyncio
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.utils.openai_functions import (
    convert_pydantic_to_openai_function,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


from traceloop.sdk import Traceloop

Traceloop.init(app_name="lcel_example")


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


async def chain():
    openai_functions = [convert_pydantic_to_openai_function(Joke)]

    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are helpful assistant"), ("user", "{input}")]
    )
    model = ChatOpenAI(model="gpt-3.5-turbo")
    output_parser = JsonOutputFunctionsParser()

    chain = prompt | model.bind(functions=openai_functions) | output_parser
    return await chain.ainvoke(
        {"input": "tell me a short joke"}, {"metadata": {"user_id": "1234"}}
    )


print(asyncio.run(chain()))
