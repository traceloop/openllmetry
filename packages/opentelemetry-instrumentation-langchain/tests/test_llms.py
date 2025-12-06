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
from langchain_openai import ChatOpenAI, OpenAI
from opentelemetry.sdk._logs import LogData
from opentelemetry.sdk.trace import Span
from opentelemetry.semconv._incubating.attributes import (
    event_attributes as EventAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace.propagation import (
    get_current_span,
)
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)
from pydantic import BaseModel, Field


def open_ai_prompt():
    # flake8: noqa: E501
    # This prompt is long on purpose to trigger the cache (in order to reproduce the caching behavior when rewriting the cassette, run it twice)
    return """
OpenTelemetry: A Deep Dive into the Standard for Observability
OpenTelemetry is an open-source project under the Cloud Native Computing Foundation (CNCF) that provides a unified set of APIs, libraries, agents, and instrumentation to enable observability—specifically tracing, metrics, and logs—in distributed systems. It was formed through the merger of two earlier CNCF projects: OpenTracing and OpenCensus, both of which had overlapping goals but different implementations and user bases. OpenTelemetry combines the best of both, aiming to create a single, vendor-agnostic standard for telemetry data collection.
Background and Motivation
Modern software systems are increasingly composed of microservices, often deployed in cloud-native environments. These distributed architectures bring significant operational complexity: services may scale dynamically, instances may be short-lived, and failures may not be obvious. As a result, understanding how systems behave in production requires powerful observability tools.
Historically, developers had to integrate separate tools for logging, metrics, and tracing, often using vendor-specific SDKs. This led to inconsistent data, vendor lock-in, and high maintenance costs. OpenTelemetry addresses this by offering a standardized, portable, and extensible approach to collecting telemetry data.
Core Goals of OpenTelemetry
Unified Observability Framework
OpenTelemetry provides a single set of APIs and libraries to collect traces, metrics, and logs, promoting consistency across services and languages.
Vendor-Neutral and Open Standards
It enables instrumentation that is decoupled from any specific observability backend. This makes it easy to switch or support multiple backends like Prometheus, Jaeger, Zipkin, or commercial tools like Datadog, New Relic, and Honeycomb.
Automatic and Manual Instrumentation
OpenTelemetry supports both automatic instrumentation—where telemetry is collected with little to no code changes—and manual instrumentation for custom spans, metrics, and logs.
Support for Multiple Languages
The project supports most major programming languages including Java, Go, Python, JavaScript/TypeScript, .NET, C++, and more.
Pluggable Architecture
The design of OpenTelemetry is modular, allowing developers to plug in their exporters, processors, and samplers, and tailor the telemetry pipeline to suit their needs.
Key Concepts and Components
1. Traces and Spans
Tracing is used to understand the flow of requests through a system. In OpenTelemetry:
A trace represents the complete journey of a request across services.
A span is a single operation within that journey (e.g., a database call, an HTTP request). Each span includes metadata like name, duration, parent-child relationships, and attributes.
OpenTelemetry uses the W3C Trace Context standard to propagate context across service boundaries, enabling distributed tracing.
2. Metrics
Metrics capture numerical data about a system’s behavior, often for performance monitoring. OpenTelemetry supports:
Counters: for counting occurrences of events.
Gauges: for measuring values at a point in time.
Histograms: for measuring distributions of values (e.g., request durations).
Metrics are collected periodically and can be aggregated and exported efficiently.
3. Logs (Logging Signals)
OpenTelemetry is actively working to bring logging into the same ecosystem, creating semantic conventions and correlation mechanisms so logs can be linked with traces and metrics. The vision is to allow logs to be structured, contextualized, and used in conjunction with the other signals.
4. Context Propagation
The Context is a core abstraction that carries state between different parts of a distributed system. OpenTelemetry provides mechanisms to extract and inject context from and into messages (HTTP headers, gRPC metadata, etc.), ensuring trace continuity across services.
The OpenTelemetry SDK Architecture
The OpenTelemetry SDK is the implementation of the API and includes the full telemetry pipeline. Its key parts include:
Instrumentation Libraries: Provide pre-built hooks for popular frameworks (e.g., Express.js, Spring, Django).
API: Language-specific interfaces to create and manipulate telemetry data.
SDK: Implements the core telemetry pipeline.
Exporter: Sends data to a backend (e.g., OTLP, Jaeger, Prometheus).
Processor: Processes telemetry data before export, such as batching or filtering.
Sampler: Determines which traces or spans are collected.
OTLP: OpenTelemetry Protocol
OpenTelemetry defines its own protocol—OTLP (OpenTelemetry Protocol)—for exporting telemetry data. OTLP supports gRPC and HTTP/Protobuf and is designed for performance, extensibility, and interoperability.
Collector
The OpenTelemetry Collector is a key component that runs as an agent or gateway between applications and telemetry backends. It has no vendor dependencies and supports:
Receivers: To receive telemetry data.
Processors: To manipulate data (filtering, batching, transformation).
Exporters: To send data to one or more observability backends.
The Collector makes it easier to centralize telemetry processing, manage data pipelines, and enforce policies like sampling or redaction.
Semantic Conventions
OpenTelemetry defines semantic conventions for common operations and attributes. These conventions ensure consistency in telemetry data across libraries, vendors, and teams. For example, HTTP spans will have standardized attributes like http.method, http.status_code, http.route, etc.
This standardization is critical for making dashboards, alerts, and queries portable and meaningful.
Language Support
OpenTelemetry has a broad ecosystem of language SDKs, each maturing at a slightly different pace. For instance:
Java and Go: Among the most mature and production-ready.
Python, .NET, JavaScript: Actively developed with good community support.
C++, PHP, Ruby: Available but may have partial support or fewer features.
Each language SDK supports the core signals (traces and metrics), with logs being integrated as work progresses.
Adoption and Ecosystem
OpenTelemetry is backed by all major cloud providers, observability vendors, and many large enterprises. It's quickly becoming the default instrumentation standard for open-source and commercial software. Projects and services like Kubernetes, Istio, Envoy, gRPC, and many frameworks are adopting OpenTelemetry natively.
The CNCF landscape now includes OpenTelemetry as a graduated project (as of 2024), reflecting its stability, maturity, and widespread usage.
Benefits for Developers and Operators
Reduced Vendor Lock-in: Instrument once, export anywhere.
Improved Developer Productivity: Consistent APIs and tooling.
Better System Understanding: Correlate logs, traces, and metrics to resolve incidents faster.
Cost Optimization: Fine-grained control over data volume and sampling.
Compliance and Security: Centralized control over telemetry pipelines.
Future Directions
OpenTelemetry’s roadmap includes:
Improved support for logs, including unified correlation with traces and metrics.
Better semantic conventions and user-defined schemas.
AI/ML for telemetry enrichment, anomaly detection, and intelligent sampling.
Context-aware observability through automatic context propagation in async/streaming environments.
Profiling signals integration in the long-term future.
Conclusion
OpenTelemetry is not just a tool or a library—it’s a movement toward a better way to understand and manage modern software systems. With deep community support, growing enterprise adoption, and a commitment to open standards, it is poised to be the foundation for observability for the next decade.
Whether you are building a single microservice or running a complex mesh of applications across hybrid clouds, OpenTelemetry offers the instrumentation, tools, and ecosystem needed to bring clarity to your system's behavior—without the downsides of proprietary lock-in or fragmented tooling.
    """


@pytest.mark.vcr
def test_custom_llm(instrument_legacy, span_exporter, log_exporter):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("user", "{input}")]
    )
    model = HuggingFaceTextGenInference(
        inference_server_url="https://w8qtunpthvh1r7a0.us-east-1.aws.endpoints.huggingface.cloud"
    )

    chain = prompt | model
    response = chain.invoke({"input": "tell me a short joke"})

    spans = span_exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.task",
        "HuggingFaceTextGenInference.completion",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    hugging_face_span = next(
        span for span in spans if span.name == "HuggingFaceTextGenInference.completion"
    )

    assert hugging_face_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"
    assert hugging_face_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "unknown"
    assert hugging_face_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "HuggingFace"
    assert (
        hugging_face_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "System: You are a helpful assistant\nHuman: tell me a short joke"
    )
    assert (
        hugging_face_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.0.content"]
        == response
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
def test_custom_llm_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("user", "{input}")]
    )
    model = HuggingFaceTextGenInference(
        inference_server_url="https://w8qtunpthvh1r7a0.us-east-1.aws.endpoints.huggingface.cloud"
    )

    chain = prompt | model
    response = chain.invoke({"input": "tell me a short joke"})

    spans = span_exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.task",
        "HuggingFaceTextGenInference.completion",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    hugging_face_span = next(
        span for span in spans if span.name == "HuggingFaceTextGenInference.completion"
    )

    assert hugging_face_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"
    assert hugging_face_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "unknown"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    # With the custom llm, LangChain is emitting only one "on_llm_start" callback,
    # because of this, both the systm instruction and the user message are in the same event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": "System: You are a helpful assistant\nHuman: tell me a short joke"},
    )

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)  # logs[1] may not exist


@pytest.mark.vcr
def test_custom_llm_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("user", "{input}")]
    )
    model = HuggingFaceTextGenInference(
        inference_server_url="https://w8qtunpthvh1r7a0.us-east-1.aws.endpoints.huggingface.cloud"
    )

    chain = prompt | model
    response = chain.invoke({"input": "tell me a short joke"})

    spans = span_exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.task",
        "HuggingFaceTextGenInference.completion",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    hugging_face_span = next(
        span for span in spans if span.name == "HuggingFaceTextGenInference.completion"
    )

    assert hugging_face_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "completion"
    assert hugging_face_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "unknown"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)  # logs[1] may not exist


@pytest.mark.vcr
def test_openai(instrument_legacy, span_exporter, log_exporter):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("human", "{input}")]
    )
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model

    # Refactored the big prompt to a function to easily duplicate the test
    prompt = open_ai_prompt()
    response = chain.invoke({"input": prompt})

    spans = span_exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.task",
        "ChatOpenAI.chat",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    openai_span = next(span for span in spans if span.name == "ChatOpenAI.chat")

    assert openai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert openai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    assert openai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"
    assert (
        (openai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"])
        == "You are a helpful assistant"
    )
    assert (openai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "system"
    assert (openai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"]) == prompt
    assert (openai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"]) == "user"

    assert openai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 1497
    assert openai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 1037
    assert openai_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 2534

    workflow_span = next(
        span for span in spans if span.name == "RunnableSequence.workflow"
    )
    output = json.loads(
        workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    # Validate the completion content via workflow output
    assert output["outputs"]["kwargs"]["content"] == response.content
    assert output["outputs"]["kwargs"]["type"] == "ai"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
def test_openai_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("human", "{input}")]
    )
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model

    # Refactored the big prompt to a function to easily duplicate the test
    prompt = open_ai_prompt()
    response = chain.invoke({"input": prompt})

    spans = span_exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.task",
        "ChatOpenAI.chat",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    openai_span = next(span for span in spans if span.name == "ChatOpenAI.chat")

    assert openai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert openai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"

    assert openai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 1497
    assert openai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 1037
    assert openai_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 2534

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    assert_message_in_logs(
        logs[0], "gen_ai.system.message", {"content": "You are a helpful assistant"}
    )

    # Validate user message Event
    assert_message_in_logs(logs[1], "gen_ai.user.message", {"content": prompt})

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response.content},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)  # logs[2] does not exist


@pytest.mark.vcr
def test_openai_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("human", "{input}")]
    )
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model

    # Refactored the big prompt to a function to easily duplicate the test
    prompt = open_ai_prompt()
    chain.invoke({"input": prompt})

    spans = span_exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.task",
        "ChatOpenAI.chat",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    openai_span = next(span for span in spans if span.name == "ChatOpenAI.chat")

    assert openai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert openai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"

    assert openai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 1497
    assert openai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 1037
    assert openai_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 2534

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    assert_message_in_logs(logs[0], "gen_ai.system.message", {})

    # Validate user message Event
    assert_message_in_logs(logs[1], "gen_ai.user.message", {})

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)  # logs[2] does not exist


@pytest.mark.vcr
def test_openai_functions(instrument_legacy, span_exporter, log_exporter):
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

    spans = span_exporter.get_finished_spans()

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
    assert openai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-3.5-turbo"
    assert (
        (openai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"])
        == "You are helpful assistant"
    )
    assert (openai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "system"
    assert (
        (openai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"])
        == "tell me a short joke"
    )
    assert (openai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"]) == "user"
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
    assert openai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 76
    assert openai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 35
    assert openai_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 111

    workflow_span = next(
        span for span in spans if span.name == "RunnableSequence.workflow"
    )
    output = json.loads(
        workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    )
    # Validate the tool call via workflow output
    assert output["outputs"] == response

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
def test_openai_functions_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()

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
    assert openai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-3.5-turbo"

    assert openai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 76
    assert openai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 35
    assert openai_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 111

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    assert_message_in_logs(
        logs[0], "gen_ai.system.message", {"content": "You are helpful assistant"}
    )

    # Validate user message Event
    assert_message_in_logs(
        logs[1], "gen_ai.user.message", {"content": "tell me a short joke"}
    )

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "function_call",
        "message": {"content": ""},
        "tool_calls": [
            {
                "function": {
                    "arguments": '{"setup":"Why did the scarecrow win an award?","punchline":"Because he was outstanding in his field!"}',
                    "name": "Joke",
                },
                "id": "",
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)  # logs[2] does not exist


@pytest.mark.vcr
def test_openai_functions_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()

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
    assert openai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-3.5-turbo"

    assert openai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 76
    assert openai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 35
    assert openai_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 111

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    assert_message_in_logs(logs[0], "gen_ai.system.message", {})

    # Validate user message Event
    assert_message_in_logs(logs[1], "gen_ai.user.message", {})

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "function_call",
        "message": {},
        "tool_calls": [{"function": {"name": "Joke"}, "id": "", "type": "function"}],
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)  # logs[2] does not exist


@pytest.mark.vcr
def test_anthropic(instrument_legacy, span_exporter, log_exporter):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("user", "{input}")]
    )
    model = ChatAnthropic(model="claude-2.1", temperature=0.5)

    chain = prompt | model
    response = chain.invoke({"input": "tell me a short joke"})

    spans = span_exporter.get_finished_spans()

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
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "claude-2.1"
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "Anthropic"
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert (
        (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"])
        == "You are a helpful assistant"
    )
    assert (
        (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "system"
    )
    assert (
        (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"])
        == "tell me a short joke"
    )
    assert (anthropic_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"]) == "user"
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 19
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 22
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 41
    assert (
        anthropic_span.attributes["gen_ai.response.id"]
        == "msg_017fMG9SRDFTBhcD1ibtN1nK"
    )
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
            "model_name": "claude-2.1",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
                "input_tokens": 19,
                "output_tokens": 22,
                "server_tool_use": None,
            },
        },
        "tool_calls": [],
        "type": "ai",
        "usage_metadata": {
            "input_token_details": {},
            "input_tokens": 19,
            "output_tokens": 22,
            "total_tokens": 41,
        },
    }

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
def test_anthropic_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("user", "{input}")]
    )
    model = ChatAnthropic(model="claude-2.1", temperature=0.5)

    chain = prompt | model
    response = chain.invoke({"input": "tell me a short joke"})

    spans = span_exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.task",
        "ChatAnthropic.chat",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    anthropic_span = next(span for span in spans if span.name == "ChatAnthropic.chat")

    assert anthropic_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "claude-2.1"
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5

    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 19
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 22
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 41
    assert (
        anthropic_span.attributes["gen_ai.response.id"]
        == "msg_017fMG9SRDFTBhcD1ibtN1nK"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    assert_message_in_logs(
        logs[0], "gen_ai.system.message", {"content": "You are a helpful assistant"}
    )

    # Validate user message Event
    assert_message_in_logs(
        logs[1], "gen_ai.user.message", {"content": "tell me a short joke"}
    )

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response.content},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)  # logs[2] does not exist


@pytest.mark.vcr
def test_anthropic_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("user", "{input}")]
    )
    model = ChatAnthropic(model="claude-2.1", temperature=0.5)

    chain = prompt | model
    chain.invoke({"input": "tell me a short joke"})

    spans = span_exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.task",
        "ChatAnthropic.chat",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    anthropic_span = next(span for span in spans if span.name == "ChatAnthropic.chat")

    assert anthropic_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "claude-2.1"
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5

    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 19
    assert anthropic_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 22
    assert anthropic_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 41
    assert (
        anthropic_span.attributes["gen_ai.response.id"]
        == "msg_017fMG9SRDFTBhcD1ibtN1nK"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    assert_message_in_logs(logs[0], "gen_ai.system.message", {})

    # Validate user message Event
    assert_message_in_logs(logs[1], "gen_ai.user.message", {})

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)  # logs[2] does not exist


@pytest.mark.vcr
def test_bedrock(instrument_legacy, span_exporter, log_exporter):
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

    spans = span_exporter.get_finished_spans()

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
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "anthropic.claude-3-haiku-20240307-v1:0"
    )
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "AWS"
    assert (
        (bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"])
        == "You are a helpful assistant"
    )
    assert (bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"]) == "system"
    assert (
        (bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.content"])
        == "tell me a short joke"
    )
    assert (bedrock_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"]) == "user"
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 16
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 27
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

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
def test_bedrock_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.task",
        "ChatBedrock.chat",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    bedrock_span = next(span for span in spans if span.name == "ChatBedrock.chat")

    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "anthropic.claude-3-haiku-20240307-v1:0"
    )

    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 16
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 27
    assert bedrock_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 43

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    assert_message_in_logs(
        logs[0], "gen_ai.system.message", {"content": "You are a helpful assistant"}
    )

    # Validate user message Event
    assert_message_in_logs(
        logs[1], "gen_ai.user.message", {"content": "tell me a short joke"}
    )

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response.content},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)  # logs[2] does not exist


@pytest.mark.vcr
def test_bedrock_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
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
    chain.invoke({"input": "tell me a short joke"})

    spans = span_exporter.get_finished_spans()

    assert [
        "ChatPromptTemplate.task",
        "ChatBedrock.chat",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    bedrock_span = next(span for span in spans if span.name == "ChatBedrock.chat")

    assert bedrock_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        bedrock_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "anthropic.claude-3-haiku-20240307-v1:0"
    )
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 16
    assert bedrock_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 27
    assert bedrock_span.attributes[SpanAttributes.LLM_USAGE_TOTAL_TOKENS] == 43

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    assert_message_in_logs(logs[0], "gen_ai.system.message", {})

    # Validate user message Event
    assert_message_in_logs(logs[1], "gen_ai.user.message", {})

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)  # logs[2] does not exist


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
def test_trace_propagation(instrument_legacy, span_exporter, log_exporter, LLM):
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

    spans = span_exporter.get_finished_spans()
    openai_span = next(span for span in spans if "OpenAI" in span.name)

    expected_vendors = {
        OpenAI: "openai",
        VLLMOpenAI: "openai", 
        ChatOpenAI: "openai"
    }
    assert openai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == expected_vendors[LLM]

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
@pytest.mark.parametrize("LLM", [OpenAI, VLLMOpenAI, ChatOpenAI])
def test_trace_propagation_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, LLM
):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant "), ("human", "{input}")]
    )
    model = LLM(
        model="facebook/opt-125m", base_url="http://localhost:8000/v1", max_tokens=20
    )
    chain = prompt | model

    send_spy = spy_decorator(httpx.Client.send)
    with patch.object(httpx.Client, "send", send_spy):
        response = chain.invoke({"input": "Tell me a joke about OpenTelemetry"})
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    openai_span = next(span for span in spans if "OpenAI" in span.name)

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)

    logs = log_exporter.get_finished_logs()

    if issubclass(LLM, ChatOpenAI):
        assert len(logs) == 3

        # Validate system message Event
        assert_message_in_logs(
            logs[0],
            "gen_ai.system.message",
            {"content": "You are a helpful assistant "},
        )

        # Validate user message Event
        assert_message_in_logs(
            logs[1],
            "gen_ai.user.message",
            {"content": "Tell me a joke about OpenTelemetry"},
        )

        # Validate AI choice Event
        choice_event = {
            "index": 0,
            "finish_reason": "length",
            "message": {"content": response.content},
        }
        assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)  # logs[2] does not exist
    else:
        assert len(logs) == 2

        # Validate system and user message Event

        # With both OpenAI and VLLMOpenAI, LangChain is emitting only one
        # "on_llm_start" callback, because of this, both the system
        # instruction and the user message are in the same event
        assert_message_in_logs(
            logs[0],
            "gen_ai.user.message",
            {
                "content": "System: You are a helpful assistant \nHuman: Tell me a joke about OpenTelemetry",
            },
        )

        # Validate AI choice Event
        choice_event = {
            "index": 0,
            "finish_reason": "length",
            "message": {"content": response},
        }
        assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)  # logs[1] may not exist


@pytest.mark.vcr
@pytest.mark.parametrize("LLM", [OpenAI, VLLMOpenAI, ChatOpenAI])
def test_trace_propagation_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, LLM
):
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

    spans = span_exporter.get_finished_spans()
    openai_span = next(span for span in spans if "OpenAI" in span.name)

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)

    logs = log_exporter.get_finished_logs()
    if issubclass(LLM, ChatOpenAI):
        assert len(logs) == 3

        # Validate system message Event
        assert_message_in_logs(logs[0], "gen_ai.system.message", {})

        # Validate user message Event
        assert_message_in_logs(logs[1], "gen_ai.user.message", {})

        # Validate AI choice Event
        choice_event = {
            "index": 0,
            "finish_reason": "length",
            "message": {},
        }
        assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)  # logs[2] does not exist
    else:
        assert len(logs) == 2

        # Validate system and user message Event

        # With both OpenAI and VLLMOpenAI, LangChain is emitting only one
        # "on_llm_start" callback, because of this, both the system
        # instruction and the user message are in the same event
        assert_message_in_logs(logs[0], "gen_ai.user.message", {})

        # Validate AI choice Event
        choice_event = {
            "index": 0,
            "finish_reason": "length",
            "message": {},
        }
        assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)  # logs[1] may not exist


@pytest.mark.vcr
@pytest.mark.parametrize(
    "LLM", [OpenAI, VLLMOpenAI, pytest.param(ChatOpenAI, marks=pytest.mark.xfail)]
)
def test_trace_propagation_stream(instrument_legacy, span_exporter, log_exporter, LLM):
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

    spans = span_exporter.get_finished_spans()
    openai_span = next(span for span in spans if "OpenAI" in span.name)

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.vcr
@pytest.mark.parametrize(
    "LLM", [OpenAI, VLLMOpenAI, pytest.param(ChatOpenAI, marks=pytest.mark.xfail)]
)
def test_trace_propagation_stream_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, LLM
):
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
        chunks = [s for s in stream]
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    openai_span = next(span for span in spans if "OpenAI" in span.name)

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate system and user message Event

    # With both OpenAI and VLLMOpenAI, LangChain is emitting only one
    # "on_llm_start" callback, because of this, both the system
    # instruction and the user message are in the same event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {
            "content": "System: You are a helpful assistant \nHuman: Tell me a joke about OpenTelemetry",
        },
    )

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "length",
        "message": {"content": "".join(chunks)},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)  # logs[1] may not exist


@pytest.mark.vcr
@pytest.mark.parametrize(
    "LLM", [OpenAI, VLLMOpenAI, pytest.param(ChatOpenAI, marks=pytest.mark.xfail)]
)
def test_trace_propagation_stream_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, LLM
):
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

    spans = span_exporter.get_finished_spans()
    openai_span = next(span for span in spans if "OpenAI" in span.name)

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate system and user message Event

    # With both OpenAI and VLLMOpenAI, LangChain is emitting only one
    # "on_llm_start" callback, because of this, both the system
    # instruction and the user message are in the same event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "length",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)  # logs[1] may not exist


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.parametrize("LLM", [OpenAI, VLLMOpenAI, ChatOpenAI])
async def test_trace_propagation_async(
    instrument_legacy, span_exporter, log_exporter, LLM
):
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

    spans = span_exporter.get_finished_spans()
    openai_span = next(span for span in spans if "OpenAI" in span.name)

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.parametrize("LLM", [OpenAI, VLLMOpenAI, ChatOpenAI])
async def test_trace_propagation_async_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, LLM
):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant "), ("human", "{input}")]
    )
    model = LLM(
        model="facebook/opt-125m", base_url="http://localhost:8000/v1", max_tokens=20
    )
    chain = prompt | model

    send_spy = spy_decorator(httpx.AsyncClient.send)
    with patch.object(httpx.AsyncClient, "send", send_spy):
        response = await chain.ainvoke({"input": "Tell me a joke about OpenTelemetry"})
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    openai_span = next(span for span in spans if "OpenAI" in span.name)

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)

    logs = log_exporter.get_finished_logs()
    if issubclass(LLM, ChatOpenAI):
        assert len(logs) == 3

        # Validate system message Event
        assert_message_in_logs(
            logs[0],
            "gen_ai.system.message",
            {"content": "You are a helpful assistant "},
        )

        # Validate user message Event
        assert_message_in_logs(
            logs[1],
            "gen_ai.user.message",
            {"content": "Tell me a joke about OpenTelemetry"},
        )

        # Validate AI choice Event
        choice_event = {
            "index": 0,
            "finish_reason": "length",
            "message": {
                "content": response.content,
            },
        }
        assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)  # logs[2] does not exist
    else:
        assert len(logs) == 2

        # Validate system and user message Event

        # With both OpenAI and VLLMOpenAI, LangChain is emitting only one
        # "on_llm_start" callback, because of this, both the system
        # instruction and the user message are in the same event
        assert_message_in_logs(
            logs[0],
            "gen_ai.user.message",
            {
                "content": "System: You are a helpful assistant \nHuman: Tell me a joke about OpenTelemetry",
            },
        )

        # Validate AI choice Event
        choice_event = {
            "index": 0,
            "finish_reason": "length",
            "message": {"content": response},
        }
        assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)  # logs[1] may not exist


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.parametrize("LLM", [OpenAI, VLLMOpenAI, ChatOpenAI])
async def test_trace_propagation_async_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, LLM
):
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

    spans = span_exporter.get_finished_spans()
    openai_span = next(span for span in spans if "OpenAI" in span.name)

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)

    logs = log_exporter.get_finished_logs()
    if issubclass(LLM, ChatOpenAI):
        assert len(logs) == 3

        # Validate system message Event
        assert_message_in_logs(logs[0], "gen_ai.system.message", {})

        # Validate user message Event
        assert_message_in_logs(logs[1], "gen_ai.user.message", {})

        # Validate AI choice Event
        choice_event = {
            "index": 0,
            "finish_reason": "length",
            "message": {},
        }
        assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)  # logs[2] does not exist
    else:
        assert len(logs) == 2

        # Validate system and user message Event

        # With both OpenAI and VLLMOpenAI, LangChain is emitting only one
        # "on_llm_start" callback, because of this, both the system
        # instruction and the user message are in the same event
        assert_message_in_logs(logs[0], "gen_ai.user.message", {})

        # Validate AI choice Event
        choice_event = {
            "index": 0,
            "finish_reason": "length",
            "message": {},
        }
        assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)  # logs[1] may not exist


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.parametrize(
    "LLM", [OpenAI, VLLMOpenAI, pytest.param(ChatOpenAI, marks=pytest.mark.xfail)]
)
async def test_trace_propagation_stream_async(
    instrument_legacy, span_exporter, log_exporter, LLM
):
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

    spans = span_exporter.get_finished_spans()
    openai_span = next(span for span in spans if "OpenAI" in span.name)

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 0, (
        "Assert that it doesn't emit logs when use_legacy_attributes is True"
    )


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.parametrize(
    "LLM", [OpenAI, VLLMOpenAI, pytest.param(ChatOpenAI, marks=pytest.mark.xfail)]
)
async def test_trace_propagation_stream_async_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, LLM
):
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
        chunks = [s async for s in stream]
    send_spy.mock.assert_called_once()

    spans = span_exporter.get_finished_spans()
    openai_span = next(span for span in spans if "OpenAI" in span.name)

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate system and user message Event

    # With both OpenAI and VLLMOpenAI, LangChain is emitting only one
    # "on_llm_start" callback, because of this, both the system
    # instruction and the user message are in the same event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {
            "content": "System: You are a helpful assistant \nHuman: Tell me a joke about OpenTelemetry",
        },
    )

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "length",
        "message": {"content": "".join(chunks)},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)  # logs[1] may not exist


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.parametrize(
    "LLM", [OpenAI, VLLMOpenAI, pytest.param(ChatOpenAI, marks=pytest.mark.xfail)]
)
async def test_trace_propagation_stream_async_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, LLM
):
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

    spans = span_exporter.get_finished_spans()
    openai_span = next(span for span in spans if "OpenAI" in span.name)

    args, kwargs = send_spy.mock.call_args
    request = args[0]

    assert_request_contains_tracecontext(request, openai_span)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate system and user message Event

    # With both OpenAI and VLLMOpenAI, LangChain is emitting only one
    # "on_llm_start" callback, because of this, both the system
    # instruction and the user message are in the same event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate AI choice Event
    choice_event = {
        "index": 0,
        "finish_reason": "length",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)  # logs[1] may not exist


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "langchain"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
