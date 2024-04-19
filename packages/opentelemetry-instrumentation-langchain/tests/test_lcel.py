import pytest
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.utils.openai_functions import (
    convert_pydantic_to_openai_function,
)
from langchain_community.llms.huggingface_text_gen_inference import (
    HuggingFaceTextGenInference,
)
from langchain_community.chat_models import BedrockChat, ChatOpenAI, ChatAnthropic
from langchain_core.pydantic_v1 import BaseModel, Field
import boto3


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
        "openai.chat",
        "ChatOpenAI.langchain.task",
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
        span for span in spans if span.name == "ChatOpenAI.langchain.task"
    )
    output_parser_task_span = next(
        span
        for span in spans
        if span.name == "JsonOutputFunctionsParser.langchain.task"
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
            "PromptTemplate.langchain.task",
            "openai.chat",
            "ChatOpenAI.langchain.task",
            "StrOutputParser.langchain.task",
            "RunnableSequence.langchain.workflow",
        ]
    ) == set([span.name for span in spans])

    workflow_span = next(
        span for span in spans if span.name == "RunnableSequence.langchain.workflow"
    )
    chat_openai_task_span = next(
        span for span in spans if span.name == "ChatOpenAI.langchain.task"
    )
    output_parser_task_span = next(
        span for span in spans if span.name == "StrOutputParser.langchain.task"
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
        "PromptTemplate.langchain.task",
        "openai.chat",
        "ChatOpenAI.langchain.task",
        "StrOutputParser.langchain.task",
        "RunnableSequence.langchain.workflow",
    ] == [span.name for span in spans]


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
        "HuggingFaceTextGenInference.chat",
        "RunnableSequence.langchain.workflow",
    ] == [span.name for span in spans]

    hugging_face_span = next(
        span for span in spans if span.name == "HuggingFaceTextGenInference.chat"
    )

    assert hugging_face_span.attributes["llm.request.type"] == "completion"
    assert (
        hugging_face_span.attributes["llm.request.model"]
        == "HuggingFaceTextGenInference"
    )
    assert (
        hugging_face_span.attributes["llm.prompts.0.user"]
        == "System: You are a helpful assistant\nHuman: tell me a short joke"
    )
    assert hugging_face_span.attributes["llm.completions.0.content"] == response


@pytest.mark.vcr
def test_openai(exporter):
    model = ChatOpenAI(model="gpt-3.5-turbo")

    response = model.generate(
        messages=[
            [
                SystemMessage(content="You are a helpful assistant"),
                HumanMessage(content="Tell me a joke about OpenTelemetry"),
            ]
        ]
    )

    spans = exporter.get_finished_spans()

    assert [
        "openai.chat",
        "ChatOpenAI.langchain.task",
    ] == [span.name for span in spans]

    openai_span = next(
        span for span in spans if span.name == "ChatOpenAI.langchain.task"
    )

    assert openai_span.attributes["llm.request.type"] == "chat"
    assert openai_span.attributes["llm.request.model"] == "gpt-3.5-turbo"
    assert (
        openai_span.attributes["llm.prompts.0.content"] == "You are a helpful assistant"
    )
    assert openai_span.attributes["llm.prompts.0.role"] == "system"
    assert (
        openai_span.attributes["llm.prompts.1.content"]
        == "Tell me a joke about OpenTelemetry"
    )
    assert openai_span.attributes["llm.prompts.1.role"] == "user"
    assert (
        openai_span.attributes["llm.completions.0.content"]
        == response.generations[0][0].text
    )


@pytest.mark.vcr
def test_anthropic(exporter):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("user", "{input}")]
    )
    model = ChatAnthropic(model="claude-2")

    chain = prompt | model
    response = chain.invoke({"input": "tell me a short joke"})

    spans = exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.langchain.task",
        "ChatAnthropic.langchain.task",
        "RunnableSequence.langchain.workflow",
    ] == [span.name for span in spans]

    anthropic_span = next(
        span for span in spans if span.name == "ChatAnthropic.langchain.task"
    )

    assert anthropic_span.attributes["llm.request.type"] == "chat"
    assert anthropic_span.attributes["llm.request.model"] == "claude-2"
    assert (
        anthropic_span.attributes["llm.prompts.0.content"]
        == "You are a helpful assistant"
    )
    assert anthropic_span.attributes["llm.completions.0.content"] == response.content


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
        "BedrockChat.langchain.task",
        "RunnableSequence.langchain.workflow",
    ] == [span.name for span in spans]

    bedrock_span = next(
        span for span in spans if span.name == "BedrockChat.langchain.task"
    )

    assert bedrock_span.attributes["llm.request.type"] == "chat"
    assert (
        bedrock_span.attributes["llm.request.model"]
        == "anthropic.claude-3-haiku-20240307-v1:0"
    )
    assert (
        bedrock_span.attributes["llm.prompts.0.content"]
        == "You are a helpful assistant"
    )
    assert bedrock_span.attributes["llm.completions.0.content"] == response.content
