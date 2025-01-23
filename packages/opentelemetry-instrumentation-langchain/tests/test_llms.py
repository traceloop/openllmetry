import json
from unittest.mock import MagicMock, patch

import boto3
import httpx
import pytest

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_community.llms.huggingface_text_gen_inference import (
    HuggingFaceTextGenInference,
)
from langchain_community.llms.vllm import VLLMOpenAI
from langchain_community.utils.openai_functions import (
    convert_pydantic_to_openai_function,
)
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAI
from opentelemetry.sdk.trace import Span
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)
from opentelemetry.trace.propagation import (
    get_current_span,
)


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
    assert openai_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 24
    assert openai_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 26
    assert openai_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 50


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

    assert set(
        [
            "ChatPromptTemplate.task",
            "JsonOutputFunctionsParser.task",
            "ChatOpenAI.chat",
            "RunnableSequence.workflow",
        ]
    ) == set([span.name for span in spans])

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
    assert openai_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 76
    assert openai_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 35
    assert openai_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 111


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
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 19
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 22
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 41
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
    model = ChatBedrock(
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
        "ChatBedrock.chat",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    bedrock_span = next(span for span in spans if span.name == "ChatBedrock.chat")
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
    assert bedrock_span.attributes[SpanAttributes.LLM_USAGE_PROMPT_TOKENS] == 16
    assert bedrock_span.attributes[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS] == 27
    assert bedrock_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 43
    output = json.loads(
        workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    # We need to remove the id from the output because it is random
    assert {k: v for k, v in output["outputs"]["kwargs"].items() if k != "id"} == {
        "content": "Here's a short joke for you:\n\nWhat do you call a bear with no teeth? A gummy bear!",
        "additional_kwargs": {
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "stop_reason": "end_turn",
            "usage": {"prompt_tokens": 16, "completion_tokens": 27, "total_tokens": 43},
        },
        "response_metadata": {
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "stop_reason": "end_turn",
            "usage": {"prompt_tokens": 16, "completion_tokens": 27, "total_tokens": 43},
        },
        "usage_metadata": {
            "input_tokens": 16,
            "output_tokens": 27,
            "total_tokens": 43,
        },
        "type": "ai",
        "tool_calls": [],
        "invalid_tool_calls": [],
    }


# from: https://stackoverflow.com/a/41599695/2749989
def spy_decorator(method_to_decorate):
    mock = MagicMock()

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        return method_to_decorate(self, *args, **kwargs)

    wrapper.mock = mock
    return wrapper


def assert_request_contains_tracecontext(request: httpx.Request, expected_span: Span):
    assert TraceContextTextMapPropagator._TRACEPARENT_HEADER_NAME in request.headers
    ctx = TraceContextTextMapPropagator().extract(request.headers)
    request_span_context = get_current_span(ctx).get_span_context()
    expected_span_context = expected_span.get_span_context()

    assert request_span_context.trace_id == expected_span_context.trace_id
    assert request_span_context.span_id == expected_span_context.span_id


@pytest.mark.vcr
@pytest.mark.parametrize("LLM", [OpenAI, VLLMOpenAI, ChatOpenAI])
def test_trace_propagation(exporter, LLM):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant "), ("human", "{input}")]
    )
    model = LLM(
        model="facebook/opt-125m", base_url="http://localhost:8000/v1", max_tokens=20
    )
    chain = prompt | model

    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        _ = chain.invoke({"input": "Tell me a joke about OpenTelemetry"})
    send_spy.mock.assert_called_once()

    spans = exporter.get_finished_spans()
    openai_span = next(span for span in spans if "OpenAI" in span.name)

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)


@pytest.mark.vcr
@pytest.mark.parametrize(
    "LLM", [OpenAI, VLLMOpenAI, pytest.param(ChatOpenAI, marks=pytest.mark.xfail)]
)
def test_trace_propagation_stream(exporter, LLM):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant "), ("human", "{input}")]
    )
    model = LLM(
        model="facebook/opt-125m", base_url="http://localhost:8000/v1", max_tokens=20
    )
    chain = prompt | model

    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        stream = chain.stream({"input": "Tell me a joke about OpenTelemetry"})
        for _ in stream:
            pass
    send_spy.mock.assert_called_once()

    spans = exporter.get_finished_spans()
    openai_span = next(span for span in spans if "OpenAI" in span.name)

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.parametrize("LLM", [OpenAI, VLLMOpenAI, ChatOpenAI])
async def test_trace_propagation_async(exporter, LLM):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant "), ("human", "{input}")]
    )
    model = LLM(
        model="facebook/opt-125m", base_url="http://localhost:8000/v1", max_tokens=20
    )
    chain = prompt | model

    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        _ = await chain.ainvoke({"input": "Tell me a joke about OpenTelemetry"})
    send_spy.mock.assert_called_once()

    spans = exporter.get_finished_spans()
    openai_span = next(span for span in spans if "OpenAI" in span.name)

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.parametrize(
    "LLM", [OpenAI, VLLMOpenAI, pytest.param(ChatOpenAI, marks=pytest.mark.xfail)]
)
async def test_trace_propagation_stream_async(exporter, LLM):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant "), ("human", "{input}")]
    )
    model = LLM(
        model="facebook/opt-125m", base_url="http://localhost:8000/v1", max_tokens=20
    )
    chain = prompt | model

    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        stream = chain.astream({"input": "Tell me a joke about OpenTelemetry"})
        async for _ in stream:
            pass
    send_spy.mock.assert_called_once()

    spans = exporter.get_finished_spans()
    openai_span = next(span for span in spans if "OpenAI" in span.name)

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)
