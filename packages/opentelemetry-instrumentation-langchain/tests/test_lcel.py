import pytest
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
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


@pytest.mark.vcr
def test_custom_llm(exporter):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are helpful assistant"), ("user", "{input}")]
    )
    model = HuggingFaceTextGenInference(
        inference_server_url="https://w8qtunpthvh1r7a0.us-east-1.aws.endpoints.huggingface.cloud"
    )

    chain = prompt | model
    response = chain.invoke({"input": "tell me a short joke"})

    spans = exporter.get_finished_spans()

    assert [
        "langchain.task.ChatPromptTemplate",
        "HuggingFaceTextGenInference.chat",
        "langchain.workflow",
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
        == "System: You are helpful assistant\nHuman: tell me a short joke"
    )
    assert hugging_face_span.attributes["llm.completions.0.content"] == response


@pytest.mark.vcr
def test_anthropic(exporter):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are helpful assistant"), ("user", "{input}")]
    )
    model = ChatAnthropic(model="claude-2")

    chain = prompt | model
    response = chain.invoke({"input": "tell me a short joke"})

    spans = exporter.get_finished_spans()

    assert [
        "langchain.task.ChatPromptTemplate",
        "langchain.task.ChatAnthropic",
        "langchain.workflow",
    ] == [span.name for span in spans]

    anthropic_span = next(
        span for span in spans if span.name == "langchain.task.ChatAnthropic"
    )

    assert anthropic_span.attributes["llm.request.type"] == "chat"
    assert anthropic_span.attributes["llm.request.model"] == "claude-2"
    assert (
        anthropic_span.attributes["llm.prompts.0.user"] == "You are helpful assistant"
    )
    assert anthropic_span.attributes["llm.completions.0.content"] == response.content


@pytest.mark.vcr
def test_bedrock(exporter):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are helpful assistant"), ("user", "{input}")]
    )
    model = BedrockChat(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        client=boto3.client('bedrock-runtime', region_name="us-east-1"),
    )

    chain = prompt | model
    response = chain.invoke({"input": "tell me a short joke"})

    spans = exporter.get_finished_spans()

    assert [
        "langchain.task.ChatPromptTemplate",
        "langchain.task.BedrockChat",
        "langchain.workflow",
    ] == [span.name for span in spans]

    bedrock_span = next(
        span for span in spans if span.name == "langchain.task.BedrockChat"
    )

    assert bedrock_span.attributes["llm.request.type"] == "chat"
    assert bedrock_span.attributes["llm.request.model"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert (
        bedrock_span.attributes["llm.prompts.0.user"] == "You are helpful assistant"
    )
    assert bedrock_span.attributes["llm.completions.0.content"] == response.content
