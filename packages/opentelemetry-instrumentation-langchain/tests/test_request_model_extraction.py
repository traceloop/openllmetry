"""
Unit tests for request/response model resolution in TraceloopCallbackHandler.

These tests do NOT use VCR cassettes; they drive the callback handler directly
with an InMemorySpanExporter, so no API keys or real HTTP are required.

Regression coverage for issue #3098: LangGraph/tool-orchestrated chat models
(e.g. init_chat_model(..., model_provider="bedrock_converse")) do not expose the
model name through invocation_params. LangChain still provides it via the callback
metadata ("ls_model_name"), so the request/response model must be resolved from
there instead of falling back to "unknown".
"""
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry import context as context_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from opentelemetry.instrumentation.langchain.callback_handler import (
    TraceloopCallbackHandler,
)

MODEL = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
# Mirrors what LangChain serializes for a ChatBedrockConverse model invoked
# through LangGraph: the class name is available, but invocation_params carry no
# model identifier.
SERIALIZED = {
    "id": ["langchain", "chat_models", "bedrock_converse", "ChatBedrockConverse"],
    "kwargs": {},
}
SPAN_NAME = "ChatBedrockConverse.chat"


@pytest.fixture
def handler_with_exporter():
    """A callback handler backed by an in-memory span exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    handler = TraceloopCallbackHandler(
        tracer=tracer,
        duration_histogram=MagicMock(),
        token_histogram=MagicMock(),
    )
    return handler, exporter


@pytest.fixture(autouse=True)
def restore_otel_context():
    """Snapshot and restore OTel context so suppression tokens cannot leak."""
    restore_token = context_api.attach(context_api.get_current())
    yield
    context_api.detach(restore_token)


def _drive_chat_call(handler, *, metadata, invocation_params):
    """Run a full chat-model lifecycle (start + end) for a single run."""
    run_id = uuid4()
    handler.on_chat_model_start(
        serialized=SERIALIZED,
        messages=[[HumanMessage(content="tell me a joke")]],
        run_id=run_id,
        metadata=metadata,
        invocation_params=invocation_params,
    )
    # bedrock_converse via LangGraph yields no model info in llm_output either,
    # so the response side must rely on the same metadata fallback.
    handler.on_llm_end(
        LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="A clean joke."))]],
            llm_output=None,
        ),
        run_id=run_id,
    )


def _finished_span(exporter):
    return next(s for s in exporter.get_finished_spans() if s.name == SPAN_NAME)


def test_request_model_resolved_from_ls_model_name(handler_with_exporter):
    """Issue #3098: request model comes from metadata when invocation_params omit it."""
    handler, exporter = handler_with_exporter
    _drive_chat_call(
        handler,
        metadata={"ls_model_name": MODEL, "ls_provider": "bedrock_converse"},
        invocation_params={"temperature": 0.0},
    )

    span = _finished_span(exporter)
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == MODEL, (
        "Issue #3098 not fixed: gen_ai.request.model is 'unknown' for LangGraph "
        "tool-orchestrated chat models. The model is available via "
        "metadata['ls_model_name'] but set_request_params never receives metadata."
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == MODEL


def test_serialized_model_takes_precedence_over_metadata(handler_with_exporter):
    """No regression: when the model is resolvable normally, metadata is not used.

    LangChain serializes the model into serialized["kwargs"] for the common case
    (e.g. ChatOpenAI -> model_name), which is what set_chat_request reads. The
    ls_model_name fallback must only kick in when that lookup yields nothing.
    """
    handler, exporter = handler_with_exporter
    run_id = uuid4()
    handler.on_chat_model_start(
        serialized={
            "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
            "kwargs": {"model_name": "gpt-4o"},
        },
        messages=[[HumanMessage(content="hi")]],
        run_id=run_id,
        metadata={"ls_model_name": "should-not-be-used"},
        invocation_params={"temperature": 0.0},
    )
    handler.on_llm_end(
        LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="ok"))]],
            llm_output=None,
        ),
        run_id=run_id,
    )

    span = next(
        s for s in exporter.get_finished_spans() if s.name == "ChatOpenAI.chat"
    )
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o"
