import pytest
from unittest.mock import patch

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv.ai import SpanAttributes, LLMRequestTypeValues
from opentelemetry.instrumentation.llamaindex.utils import (
    set_workflow_context,
    should_send_prompts,
)
from opentelemetry.instrumentation.llamaindex.config import Config

from llama_index.core.llms import ChatMessage, MessageRole, MockLLM
from llama_index.core.tools import FunctionTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import Settings
from llama_index.core.query_pipeline import InputComponent, LLMComponent, QueryPipeline

def get_span_by_name(spans: list[ReadableSpan], name: str) -> ReadableSpan:
    return next((span for span in spans if span.name == name), None)

def get_events_by_name(events: list, name: str) -> list[dict]:
    return [event for event in events if event.name == name]

@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    set_workflow_context()
    yield

@pytest.fixture(scope="module")
def llm():
    Settings.llm = MockLLM()
    return Settings.llm

def reset_llm(llm):
    llm.reset_counts()
    llm.last_message = None
    llm.last_prompt = None

class TestAgents:
    @pytest.mark.parametrize("use_legacy_attributes_fixture", [True, False])
    def test_agents_attributes_and_events(
        self, exporter, tracer, llm, use_legacy_attributes_fixture
    ):
        def add(a: int, b: int) -> int:
            """Add two integers and returns the result integer."""
            return a + b

        add_tool = FunctionTool.from_defaults(fn=add)

        agent = OpenAIAgent.from_tools(
            [add_tool],
            llm=llm,
            verbose=True,
        )

        reset_llm(llm)
        agent.chat("What is 123 + 456?")
        spans = exporter.get_finished_spans()
        span = get_span_by_name(spans, "OpenAIAgent.agent")

        if use_legacy_attributes_fixture:
            # Check for legacy attributes
            if should_send_prompts():
                assert span.attributes.get("llm.prompts.0.content") is not None
                assert span.attributes.get("llm.completions.0.content") is not None
            assert get_events_by_name(span.events, "prompt") == []
            assert get_events_by_name(span.events, "completion") == []
        else:
            # Verify prompt event
            prompt_events = get_events_by_name(span.events, "prompt")
            if should_send_prompts():
                assert len(prompt_events) >= 1
                for event in prompt_events:
                    assert "messaging.index" in event.attributes
                    assert "messaging.content" in event.attributes
                    assert "messaging.role" in event.attributes

                # Verify completion event
                completion_events = get_events_by_name(span.events, "completion")
                assert len(completion_events) >= 1
                for event in completion_events:
                    assert "messaging.index" in event.attributes
                    assert "messaging.content" in event.attributes

            assert span.attributes.get("llm.prompts.0.content") is None
            assert span.attributes.get("llm.completions.0.content") is None

class TestQueryPipelines:
    @pytest.mark.parametrize("use_legacy_attributes_fixture", [True, False])
    def test_query_pipeline_events(
        self, exporter, tracer, llm, use_legacy_attributes_fixture
    ):
        p = QueryPipeline(
            modules={
                "input": InputComponent(),
                "llm": LLMComponent(llm=llm),
            }
        )
        p.link("input", "llm")

        reset_llm(llm)
        p.run(input="What is 1 + 1?")
        spans = exporter.get_finished_spans()
        span = get_span_by_name(
            spans, "llama_index_query_pipeline.workflow"
        )  # This may need adjusting based on your actual span names

        if use_legacy_attributes_fixture:
            # Check for legacy attributes
            if should_send_prompts():
                assert span.attributes.get("llm.prompts.0.content") is not None
                assert span.attributes.get("llm.completions.0.content") is not None
            assert get_events_by_name(span.events, "prompt") == []
            assert get_events_by_name(span.events, "completion") == []
        else:
            # Verify prompt event
            prompt_events = get_events_by_name(span.events, "prompt")
            if should_send_prompts():
                assert len(prompt_events) >= 1
                for event in prompt_events:
                    assert "messaging.index" in event.attributes
                    assert "messaging.content" in event.attributes
                    assert "messaging.role" in event.attributes

                # Verify completion event
                completion_events = get_events_by_name(span.events, "completion")
                assert len(completion_events) >= 1
                for event in completion_events:
                    assert "messaging.index" in event.attributes
                    assert "messaging.content" in event.attributes

            assert span.attributes.get("llm.prompts.0.content") is None
            assert span.attributes.get("llm.completions.0.content") is None