import pytest
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.utils.openai_functions import (
    convert_pydantic_to_openai_function,
)
from langchain_community.chat_models import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field


@pytest.mark.vcr
def test_simple_lcel(exporter):
    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

    openai_functions = [convert_pydantic_to_openai_function(Joke)]

    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are helpful assistant"), ("user", "{input}")]
    )
    model = ChatOpenAI(model="gpt-3.5-turbo")
    output_parser = JsonOutputFunctionsParser()

    chain = prompt | model.bind(functions=openai_functions) | output_parser
    chain.invoke({"input": "tell me a short joke"})

    spans = exporter.get_finished_spans()

    assert [
        "langchain.task.ChatPromptTemplate",
        "openai.chat",
        "langchain.task.ChatOpenAI",
        "langchain.task.JsonOutputFunctionsParser",
        "langchain.workflow",
    ] == [span.name for span in spans]

    workflow_span = next(span for span in spans if span.name == "langchain.workflow")
    prompt_task_span = next(
        span for span in spans if span.name == "langchain.task.ChatPromptTemplate"
    )
    chat_openai_task_span = next(
        span for span in spans if span.name == "langchain.task.ChatOpenAI"
    )
    output_parser_task_span = next(
        span
        for span in spans
        if span.name == "langchain.task.JsonOutputFunctionsParser"
    )

    assert prompt_task_span.parent.span_id == workflow_span.context.span_id
    assert chat_openai_task_span.parent.span_id == workflow_span.context.span_id
    assert output_parser_task_span.parent.span_id == workflow_span.context.span_id


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_lcel(exporter):

    chat = ChatOpenAI(
        model="gpt-4",
        temperature=0,
    )

    prompt = PromptTemplate.from_template(
        "write 10 lines of random text about ${product}"
    )
    runnable = prompt | chat | StrOutputParser()
    await runnable.ainvoke({"product": "colorful socks"})

    spans = exporter.get_finished_spans()

    assert set(
        [
            "langchain.task.PromptTemplate",
            "openai.chat",
            "langchain.task.ChatOpenAI",
            "langchain.task.StrOutputParser",
            "langchain.workflow",
        ]
    ) == set([span.name for span in spans])

    workflow_span = next(span for span in spans if span.name == "langchain.workflow")
    chat_openai_task_span = next(
        span for span in spans if span.name == "langchain.task.ChatOpenAI"
    )
    output_parser_task_span = next(
        span for span in spans if span.name == "langchain.task.StrOutputParser"
    )

    assert chat_openai_task_span.parent.span_id == workflow_span.context.span_id
    assert output_parser_task_span.parent.span_id == workflow_span.context.span_id


@pytest.mark.vcr
def test_streaming(exporter):

    chat = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        streaming=True,
    )

    prompt = PromptTemplate.from_template(
        "write 10 lines of random text about ${product}"
    )
    runnable = prompt | chat | StrOutputParser()
    runnable.invoke({"product": "colorful socks"})

    spans = exporter.get_finished_spans()

    assert [
        "langchain.task.PromptTemplate",
        "openai.chat",
        "langchain.task.ChatOpenAI",
        "langchain.task.StrOutputParser",
        "langchain.workflow",
    ] == [span.name for span in spans]
