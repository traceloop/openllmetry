import json

import boto3
import pytest
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import BedrockChat
from langchain_community.llms.huggingface_text_gen_inference import (
    HuggingFaceTextGenInference,
)
from langchain_community.utils.openai_functions import (
    convert_pydantic_to_openai_function,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from opentelemetry.semconv.ai import SpanAttributes


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
        "ChatPromptTemplate.langchain.task",
        "ChatOpenAI.langchain",
        "JsonOutputFunctionsParser.langchain.task",
        "ThisIsATestChain.langchain.workflow",
    ] == [span.name for span in spans]

    workflow_span = next(
        span for span in spans if span.name == "ThisIsATestChain.langchain.workflow"
    )
    prompt_task_span = next(
        span for span in spans if span.name == "ChatPromptTemplate.langchain.task"
    )
    chat_openai_task_span = next(
        span for span in spans if span.name == "ChatOpenAI.langchain"
    )
    output_parser_task_span = next(
        span
        for span in spans
        if span.name == "JsonOutputFunctionsParser.langchain.task"
    )
    openai_span = next(span for span in spans if span.name == "openai.chat")

    assert openai_span.parent.span_id == chat_openai_task_span.context.span_id
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
    assert json.loads(
        prompt_task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    ) == {
        "outputs": "messages=[SystemMessage(content='You are helpful assistant'), "
        "HumanMessage(content='tell me a short joke')]",
        "kwargs": {"tags": ["seq:step:1", "test_tag"]},
    }


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
        "PromptTemplate.langchain.task",
        "ChatOpenAI.langchain",
        "StrOutputParser.langchain.task",
        "RunnableSequence.langchain.workflow",
    } == set([span.name for span in spans])

    workflow_span = next(
        span for span in spans if span.name == "RunnableSequence.langchain.workflow"
    )
    prompt_task_span = next(
        span for span in spans if span.name == "PromptTemplate.langchain.task"
    )
    chat_openai_task_span = next(
        span for span in spans if span.name == "ChatOpenAI.langchain"
    )
    output_parser_task_span = next(
        span for span in spans if span.name == "StrOutputParser.langchain.task"
    )
    openai_span = next(span for span in spans if span.name == "openai.chat")

    assert openai_span.parent.span_id == chat_openai_task_span.context.span_id
    assert prompt_task_span.parent.span_id == workflow_span.context.span_id
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
        "PromptTemplate.langchain.task",
        "ChatOpenAI.langchain",
        "StrOutputParser.langchain.task",
        "RunnableSequence.langchain.workflow",
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
        "PromptTemplate.langchain.task",
        "ChatOpenAI.langchain",
        "StrOutputParser.langchain.task",
        "RunnableSequence.langchain.workflow",
    ] == [span.name for span in spans]

    workflow_span = next(
        span for span in spans if span.name == "RunnableSequence.langchain.workflow"
    )
    prompt_task_span = next(
        span for span in spans if span.name == "PromptTemplate.langchain.task"
    )
    chat_openai_task_span = next(
        span for span in spans if span.name == "ChatOpenAI.langchain"
    )
    output_parser_task_span = next(
        span for span in spans if span.name == "StrOutputParser.langchain.task"
    )
    openai_span = next(span for span in spans if span.name == "openai.chat")

    assert openai_span.parent.span_id == chat_openai_task_span.context.span_id
    assert prompt_task_span.parent.span_id == workflow_span.context.span_id
    assert chat_openai_task_span.parent.span_id == workflow_span.context.span_id
    assert output_parser_task_span.parent.span_id == workflow_span.context.span_id


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
        "PromptTemplate.langchain.task",
        "ChatOpenAI.langchain",
        "StrOutputParser.langchain.task",
        "RunnableSequence.langchain.workflow",
    ] == [span.name for span in spans]

    workflow_span = next(
        span for span in spans if span.name == "RunnableSequence.langchain.workflow"
    )
    prompt_task_span = next(
        span for span in spans if span.name == "PromptTemplate.langchain.task"
    )
    chat_openai_task_span = next(
        span for span in spans if span.name == "ChatOpenAI.langchain"
    )
    output_parser_task_span = next(
        span for span in spans if span.name == "StrOutputParser.langchain.task"
    )
    openai_span = next(span for span in spans if span.name == "openai.chat")

    assert openai_span.parent.span_id == chat_openai_task_span.context.span_id
    assert prompt_task_span.parent.span_id == workflow_span.context.span_id
    assert chat_openai_task_span.parent.span_id == workflow_span.context.span_id
    assert output_parser_task_span.parent.span_id == workflow_span.context.span_id


@pytest.mark.vcr
def test_custom_llm(exporter):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("user", "{input}")]
    )
    model = HuggingFaceTextGenInference(
        inference_server_url="https://w8qtunpthvh1r7a0.us-east-1.aws.endpoints.huggingface.cloud"
    )

    chain = prompt | model
    response = chain.invoke({"input": "tell me a short joke"})

    spans = exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.langchain.task",
        "HuggingFaceTextGenInference.langchain",
        "RunnableSequence.langchain.workflow",
    ] == [span.name for span in spans]

    workflow_span = next(
        span for span in spans if span.name == "RunnableSequence.langchain.workflow"
    )
    prompt_task_span = next(
        span for span in spans if span.name == "ChatPromptTemplate.langchain.task"
    )
    hugging_face_span = next(
        span for span in spans if span.name == "HuggingFaceTextGenInference.langchain"
    )

    assert prompt_task_span.parent.span_id == workflow_span.context.span_id
    assert hugging_face_span.parent.span_id == prompt_task_span.context.span_id

    assert hugging_face_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"
    assert hugging_face_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "unknown"
    assert (
        hugging_face_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "System: You are a helpful assistant\nHuman: tell me a short joke"
    )
    assert (
        hugging_face_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == response
    )


@pytest.mark.vcr
def test_openai(exporter):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("human", "{input}")]
    )
    model = ChatOpenAI(model="gpt-3.5-turbo")
    chain = prompt | model

    response = chain.invoke({"input": "Tell me a joke about OpenTelemetry"})

    spans = exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.langchain.task",
        "ChatOpenAI.langchain",
        "RunnableSequence.langchain.workflow",
    ] == [span.name for span in spans]

    openai_span = next(span for span in spans if span.name == "ChatOpenAI.langchain")
    workflow_span = next(
        span for span in spans if span.name == "RunnableSequence.langchain.workflow"
    )
    prompt_task_span = next(
        span for span in spans if span.name == "ChatPromptTemplate.langchain.task"
    )
    chat_openai_task_span = next(
        span for span in spans if span.name == "ChatOpenAI.langchain"
    )

    assert openai_span.parent.span_id == chat_openai_task_span.context.span_id
    assert prompt_task_span.parent.span_id == workflow_span.context.span_id
    assert chat_openai_task_span.parent.span_id == workflow_span.context.span_id

    assert openai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert openai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert (
        openai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
    ) == "You are a helpful assistant"
    assert (openai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "system"
    assert (
        openai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
    ) == "Tell me a joke about OpenTelemetry"
    assert (openai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"]) == "user"
    assert (
        openai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == response.content
    )
    assert (
        openai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"]
    ) == "assistant"


@pytest.mark.vcr
def test_anthropic(exporter):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("user", "{input}")]
    )
    model = ChatAnthropic(model="claude-2.1")

    chain = prompt | model
    response = chain.invoke({"input": "tell me a short joke"})

    spans = exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.langchain.task",
        "ChatAnthropic.langchain",
        "RunnableSequence.langchain.workflow",
    ] == [span.name for span in spans]

    anthropic_span = next(
        span for span in spans if span.name == "ChatAnthropic.langchain"
    )
    workflow_span = next(
        span for span in spans if span.name == "RunnableSequence.langchain.workflow"
    )
    prompt_task_span = next(
        span for span in spans if span.name == "ChatPromptTemplate.langchain.task"
    )

    assert prompt_task_span.parent.span_id == workflow_span.context.span_id
    assert anthropic_span.parent.span_id == workflow_span.context.span_id

    assert anthropic_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert anthropic_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "claude-2.1"
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
    ) == "You are a helpful assistant"
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]
    ) == "system"
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
    ) == "tell me a short joke"
    assert (anthropic_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"]) == "user"
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == response.content
    )
    assert (
        anthropic_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"]
    ) == "assistant"
    outputs = json.loads(
        workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )["outputs"]
    assert (
        dict(re.findall(r'(\S+)=(".*?"|\S+)', outputs))["content"]
        == f'"{response.content}"'
    )


@pytest.mark.vcr
def test_bedrock(exporter):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("user", "{input}")]
    )
    model = BedrockChat(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        client=boto3.client(
            "bedrock-runtime",
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            aws_session_token="a/mock/token",
            region_name="us-east-1",
        ),
    )

    chain = prompt | model
    response = chain.invoke({"input": "tell me a short joke"})

    spans = exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.langchain.task",
        "BedrockChat.langchain",
        "RunnableSequence.langchain.workflow",
    ] == [span.name for span in spans]

    bedrock_span = next(span for span in spans if span.name == "BedrockChat.langchain")
    workflow_span = next(
        span for span in spans if span.name == "RunnableSequence.langchain.workflow"
    )
    prompt_task_span = next(
        span for span in spans if span.name == "ChatPromptTemplate.langchain.task"
    )
    completion_span = next(span for span in spans if span.name == "bedrock.completion")

    assert prompt_task_span.parent.span_id == workflow_span.context.span_id
    assert bedrock_span.parent.span_id == workflow_span.context.span_id
    assert completion_span.parent.span_id == bedrock_span.context.span_id

    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        bedrock_span.attributes[SpanAttributes.LLM_REQUEST_MODEL]
        == "anthropic.claude-3-haiku-20240307-v1:0"
    )
    assert (
        bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
    ) == "You are a helpful assistant"
    assert (bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "system"
    assert (
        bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
    ) == "tell me a short joke"
    assert (bedrock_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"]) == "user"
    assert (
        bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.content"]
        == response.content
    )
    assert (
        bedrock_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.role"]
    ) == "assistant"
    outputs = json.loads(
        workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )["outputs"]
    assert (
        dict(re.findall(r'(\S+)=(".*?"|\S+)', outputs))["content"].replace("\\n", "\n")
        == f'"{response.content}"'
    )
