import json

import boto3
import pytest
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
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
from opentelemetry.semconv_ai import SpanAttributes


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
        "ChatPromptTemplate.task",
        "HuggingFaceTextGenInference.completion",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    hugging_face_span = next(
        span for span in spans if span.name == "HuggingFaceTextGenInference.completion"
    )

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
        "ChatPromptTemplate.task",
        "ChatOpenAI.chat",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    openai_span = next(span for span in spans if span.name == "ChatOpenAI.chat")

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
def test_openai_functions(exporter):
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
    response = chain.invoke({"input": "tell me a short joke"})

    spans = exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.task",
        "ChatOpenAI.chat",
        "JsonOutputFunctionsParser.task",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    openai_span = next(span for span in spans if span.name == "ChatOpenAI.chat")

    assert openai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert openai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert (
        openai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
    ) == "You are helpful assistant"
    assert (openai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"]) == "system"
    assert (
        openai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.content"]
    ) == "tell me a short joke"
    assert (openai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"]) == "user"
    assert (
        openai_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.name"]
        == "Joke"
    )
    assert (
        openai_span.attributes[f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.description"]
        == "Joke to tell user."
    )
    assert (
        json.loads(
            openai_span.attributes[
                f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.0.parameters"
            ]
        )
    ) == {
        "type": "object",
        "properties": {
            "setup": {"description": "question to set up a joke", "type": "string"},
            "punchline": {
                "description": "answer to resolve the joke",
                "type": "string",
            },
        },
        "required": ["setup", "punchline"],
    }
    assert (
        openai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.name"]
        == "Joke"
    )
    assert (
        json.loads(
            openai_span.attributes[
                f"{SpanAttributes.LLM_COMPLETIONS}.0.tool_calls.0.arguments"
            ]
        )
        == response
    )


@pytest.mark.vcr
def test_anthropic(exporter):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("user", "{input}")]
    )
    model = ChatAnthropic(model="claude-2.1", temperature=0.5)

    chain = prompt | model
    response = chain.invoke({"input": "tell me a short joke"})

    spans = exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.task",
        "ChatAnthropic.chat",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    anthropic_span = next(span for span in spans if span.name == "ChatAnthropic.chat")
    workflow_span = next(
        span for span in spans if span.name == "RunnableSequence.workflow"
    )

    assert anthropic_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert anthropic_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "claude-2.1"
    assert anthropic_span.attributes[SpanAttributes.LLM_REQUEST_TEMPERATURE] == 0.5
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
    output = json.loads(
        workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    # We need to remove the id from the output because it is random
    assert {k: v for k, v in output["outputs"]["kwargs"].items() if k != "id"} == {
        "content": "Why can't a bicycle stand up by itself? Because it's two-tired!",
        "invalid_tool_calls": [],
        "response_metadata": {
            "id": "msg_017fMG9SRDFTBhcD1ibtN1nK",
            "model": "claude-2.1",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 19, "output_tokens": 22},
        },
        "tool_calls": [],
        "type": "ai",
        "usage_metadata": {"input_tokens": 19, "output_tokens": 22, "total_tokens": 41},
    }


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
        "ChatPromptTemplate.task",
        "BedrockChat.chat",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    bedrock_span = next(span for span in spans if span.name == "BedrockChat.chat")
    workflow_span = next(
        span for span in spans if span.name == "RunnableSequence.workflow"
    )

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
    output = json.loads(
        workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    # We need to remove the id from the output because it is random
    assert {k: v for k, v in output["outputs"]["kwargs"].items() if k != "id"} == {
        "content": "Here's a short joke for you:\n\nWhat do you call a bear with no teeth? A gummy bear!",
        "response_metadata": {
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "usage": {"prompt_tokens": 16, "completion_tokens": 27, "total_tokens": 43},
        },
        "type": "ai",
        "tool_calls": [],
        "invalid_tool_calls": [],
    }
