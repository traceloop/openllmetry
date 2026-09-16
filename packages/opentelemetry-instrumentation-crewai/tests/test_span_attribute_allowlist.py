"""
Span attributes come from a fixed set of fields, never a __dict__ walk, so an
object we don't control (an LLM client, an embedder config) can't leak its
credentials through its repr.

Objects are constructed only -- no kickoff, no network.
"""

import json

import pytest
from crewai import LLM, Agent, Crew, Task
from crewai.tools import BaseTool
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from opentelemetry.instrumentation.crewai.crewai_span_attributes import CrewAISpanAttributes

# Not a credential: an opaque marker with no provider shape, used only to prove
# that whatever is configured on an LLM or embedder stays off the span.
SENTINEL = "SENTINEL-NOT-A-KEY-9f3a"


class EchoTool(BaseTool):
    """A tool with nothing secret on it, so any leak in the span is the LLM's."""

    name: str = "echo"
    description: str = "Echoes the input back."

    def _run(self, text: str = "") -> str:
        """Echo the input back; never called, the tool is only ever serialized."""
        return text


def build_agent():
    """An Agent holding a credential on both its LLM and its embedder config."""
    return Agent(
        role="researcher",
        goal="find things",
        backstory="a fixed backstory",
        llm=LLM(model="test-model", api_key=SENTINEL),
        embedder={"provider": "openai", "config": {"api_key": SENTINEL}},
        tools=[EchoTool()],
    )


def build_task():
    """A Task whose agent holds the credential."""
    return Task(description="a fixed description", expected_output="a fixed output",
                agent=build_agent(), tools=[EchoTool()])


def build_crew():
    """A Crew holding a credential on its manager LLM and its embedder config."""
    agent = build_agent()
    return Crew(
        agents=[agent],
        tasks=[Task(description="a fixed description", expected_output="a fixed output",
                    agent=agent, tools=[EchoTool()])],
        name="fixed-crew",
        manager_llm=LLM(model="test-model", api_key=SENTINEL),
        embedder={"provider": "openai", "config": {"api_key": SENTINEL}},
    )


BUILDERS = {"Agent": build_agent, "Task": build_task, "Crew": build_crew}


def span_attributes(instance):
    """Return the attributes the instrumentation lands on a span for `instance`."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer(__name__).start_as_current_span("test") as span:
        CrewAISpanAttributes(span=span, instance=instance)
    return dict(exporter.get_finished_spans()[0].attributes)


@pytest.mark.parametrize("kind", list(BUILDERS))
def test_configured_credentials_never_reach_the_span(kind):
    """Nothing configured on an LLM or embedder is stringified onto the span."""
    attrs = span_attributes(BUILDERS[kind]())

    # Substring check: nested agents, tasks and tools are JSON-dumped into a
    # single value, so a key-wise check would miss anything hidden inside them.
    for key, value in attrs.items():
        assert SENTINEL not in str(value), f"{key} carries configured LLM state"

    # ...while the allowlisted fields are still emitted.
    assert attrs[f"crewai.{kind.lower()}.id"]


@pytest.mark.parametrize("key", ["crewai.crew.agents", "crewai.crew.tasks"])
def test_nested_tools_are_an_array_not_a_re_encoded_string(key):
    """Tools nested in the crew JSON decode in one pass, like every sibling field."""
    tools = json.loads(span_attributes(build_crew())[key])[0]["tools"]

    assert [tool["name"] for tool in tools] == ["echo"]


@pytest.mark.parametrize("kind", ["Agent", "Task"])
def test_standalone_tools_are_a_json_array_string(kind):
    """A standalone span still carries tools as a JSON array, not a Python repr."""
    tools = json.loads(span_attributes(BUILDERS[kind]())[f"crewai.{kind.lower()}.tools"])

    assert [tool["name"] for tool in tools] == ["echo"]
