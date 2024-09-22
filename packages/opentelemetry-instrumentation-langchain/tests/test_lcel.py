import json

import pytest
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.utils.openai_functions import (
    convert_pydantic_to_openai_function,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from opentelemetry.semconv_ai import SpanAttributes


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

    chain = (
        prompt | model.bind(functions=openai_functions) | output_parser
    ).with_config({"run_name": "ThisIsATestChain", "tags": ["test_tag"]})
    chain.invoke({"input": "tell me a short joke"})

    spans = exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.task",
        "ChatOpenAI.chat",
        "JsonOutputFunctionsParser.task",
        "ThisIsATestChain.workflow",
    ] == [span.name for span in spans]

    workflow_span = next(
        span for span in spans if span.name == "ThisIsATestChain.workflow"
    )
    prompt_task_span = next(
        span for span in spans if span.name == "ChatPromptTemplate.task"
    )
    chat_openai_task_span = next(
        span for span in spans if span.name == "ChatOpenAI.chat"
    )
    output_parser_task_span = next(
        span for span in spans if span.name == "JsonOutputFunctionsParser.task"
    )

    assert prompt_task_span.parent.span_id == workflow_span.context.span_id
    assert chat_openai_task_span.parent.span_id == workflow_span.context.span_id
    assert output_parser_task_span.parent.span_id == workflow_span.context.span_id

    assert json.loads(
        workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    ) == {
        "inputs": {"input": "tell me a short joke"},
        "tags": ["test_tag"],
        "metadata": {},
        "kwargs": {"name": "ThisIsATestChain"},
    }
    assert json.loads(
        workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    ) == {
        "outputs": {
            "setup": "Why couldn't the bicycle stand up by itself?",
            "punchline": "It was two tired!",
        },
        "kwargs": {"tags": ["test_tag"]},
    }
    assert json.loads(
        prompt_task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    ) == {
        "inputs": {"input": "tell me a short joke"},
        "tags": ["seq:step:1", "test_tag"],
        "metadata": {},
        "kwargs": {
            "run_type": "prompt",
            "name": "ChatPromptTemplate",
        },
    }
    assert prompt_task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT] == (
        '{"outputs": "messages=['
        "SystemMessage(content='You are helpful assistant', additional_kwargs={}, response_metadata={}),"
        " HumanMessage(content='tell me a short joke', additional_kwargs={}, response_metadata={})]\","
        ' "kwargs": {"tags": ["seq:step:1", "test_tag"]}}'
    )


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
    response = await runnable.ainvoke({"product": "colorful socks"})

    spans = exporter.get_finished_spans()

    assert {
        "PromptTemplate.task",
        "ChatOpenAI.chat",
        "StrOutputParser.task",
        "RunnableSequence.workflow",
    } == set([span.name for span in spans])

    workflow_span = next(
        span for span in spans if span.name == "RunnableSequence.workflow"
    )
    chat_openai_task_span = next(
        span for span in spans if span.name == "ChatOpenAI.chat"
    )
    output_parser_task_span = next(
        span for span in spans if span.name == "StrOutputParser.task"
    )

    assert chat_openai_task_span.parent.span_id == workflow_span.context.span_id
    assert output_parser_task_span.parent.span_id == workflow_span.context.span_id

    assert json.loads(
        workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    ) == {
        "inputs": {"product": "colorful socks"},
        "tags": [],
        "metadata": {},
        "kwargs": {"name": "RunnableSequence"},
    }
    assert json.loads(
        workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    ) == {
        "outputs": response,
        "kwargs": {"tags": []},
    }


@pytest.mark.vcr
def test_invoke(exporter):
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
        "PromptTemplate.task",
        "ChatOpenAI.chat",
        "StrOutputParser.task",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]


@pytest.mark.vcr
def test_stream(exporter):
    chat = ChatOpenAI(
        model="gpt-4",
        temperature=0,
    )

    prompt = PromptTemplate.from_template(
        "write 10 lines of random text about ${product}"
    )
    runnable = prompt | chat | StrOutputParser()
    res = runnable.stream(
        input={"product": "colorful socks"},
        config={"configurable": {"session_id": 1234}},
    )
    for _ in res:
        pass

    spans = exporter.get_finished_spans()

    assert [
        "PromptTemplate.task",
        "ChatOpenAI.chat",
        "StrOutputParser.task",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_invoke(exporter):
    chat = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        streaming=True,
    )

    prompt = PromptTemplate.from_template(
        "write 10 lines of random text about ${product}"
    )
    runnable = prompt | chat | StrOutputParser()
    await runnable.ainvoke({"product": "colorful socks"})

    spans = exporter.get_finished_spans()

    assert [
        "PromptTemplate.task",
        "ChatOpenAI.chat",
        "StrOutputParser.task",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]
