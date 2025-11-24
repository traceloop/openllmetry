import datetime
import json

import pytest
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.utils.function_calling import (
    convert_pydantic_to_openai_function,
)
from langchain_openai import ChatOpenAI
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from pydantic import BaseModel, Field


@pytest.mark.vcr
def test_simple_lcel(instrument_legacy, span_exporter, log_exporter):
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

    spans = span_exporter.get_finished_spans()

    assert set(
        [
            "ChatPromptTemplate.task",
            "JsonOutputFunctionsParser.task",
            "ChatOpenAI.chat",
            "ThisIsATestChain.workflow",
        ]
    ) == set([span.name for span in spans])

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
    assert json.loads(
        prompt_task_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT]
    ) == {
        "kwargs": {"tags": ["seq:step:1", "test_tag"]},
        "outputs": {
            "id": ["langchain", "prompts", "chat", "ChatPromptValue"],
            "kwargs": {
                "messages": [
                    {
                        "id": ["langchain", "schema", "messages", "SystemMessage"],
                        "kwargs": {
                            "content": "You are helpful " "assistant",
                            "type": "system",
                        },
                        "lc": 1,
                        "type": "constructor",
                    },
                    {
                        "id": ["langchain", "schema", "messages", "HumanMessage"],
                        "kwargs": {
                            "content": "tell me a short " "joke",
                            "type": "human",
                        },
                        "lc": 1,
                        "type": "constructor",
                    },
                ]
            },
            "lc": 1,
            "type": "constructor",
        },
    }

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_simple_lcel_with_events_with_content(
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

    chain = (
        prompt | model.bind(functions=openai_functions) | output_parser
    ).with_config({"run_name": "ThisIsATestChain", "tags": ["test_tag"]})
    chain.invoke({"input": "tell me a short joke"})

    spans = span_exporter.get_finished_spans()

    assert set(
        [
            "ChatPromptTemplate.task",
            "JsonOutputFunctionsParser.task",
            "ChatOpenAI.chat",
            "ThisIsATestChain.workflow",
        ]
    ) == set([span.name for span in spans])

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
    _choice_event = {
        "index": 0,
        "finish_reason": "function_call",
        "message": {"content": ""},
        "tool_calls": [
            {
                "id": "",
                "function": {
                    "name": "Joke",
                    "arguments": '{"setup":"Why couldn\'t the bicycle stand up by itself?","punchline":"It was two '
                    'tired!"}',
                },
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
def test_simple_lcel_with_events_with_no_content(
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

    chain = (
        prompt | model.bind(functions=openai_functions) | output_parser
    ).with_config({"run_name": "ThisIsATestChain", "tags": ["test_tag"]})
    chain.invoke({"input": "tell me a short joke"})

    spans = span_exporter.get_finished_spans()

    assert set(
        [
            "ChatPromptTemplate.task",
            "JsonOutputFunctionsParser.task",
            "ChatOpenAI.chat",
            "ThisIsATestChain.workflow",
        ]
    ) == set([span.name for span in spans])

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

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    assert_message_in_logs(logs[0], "gen_ai.system.message", {})

    # Validate user message Event
    assert_message_in_logs(logs[1], "gen_ai.user.message", {})

    # Validate AI choice Event
    _choice_event = {
        "index": 0,
        "finish_reason": "function_call",
        "message": {},
        "tool_calls": [{"function": {"name": "Joke"}, "id": "", "type": "function"}],
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_lcel(instrument_legacy, span_exporter, log_exporter):
    chat = ChatOpenAI(
        model="gpt-4",
        temperature=0,
    )

    prompt = PromptTemplate.from_template(
        "write 10 lines of random text about ${product}"
    )
    runnable = prompt | chat | StrOutputParser()
    response = await runnable.ainvoke({"product": "colorful socks"})

    spans = span_exporter.get_finished_spans()

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

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_lcel_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    chat = ChatOpenAI(
        model="gpt-4",
        temperature=0,
    )

    prompt_template = "write 10 lines of random text about ${product}"
    prompt = PromptTemplate.from_template(prompt_template)
    runnable = prompt | chat | StrOutputParser()
    response = await runnable.ainvoke({"product": "colorful socks"})

    spans = span_exporter.get_finished_spans()

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

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": prompt_template.format(product="colorful socks")},
    )

    assert response != ""

    # Validate AI choice Event
    _choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_lcel_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    chat = ChatOpenAI(
        model="gpt-4",
        temperature=0,
    )

    prompt = PromptTemplate.from_template(
        "write 10 lines of random text about ${product}"
    )
    runnable = prompt | chat | StrOutputParser()
    await runnable.ainvoke({"product": "colorful socks"})

    spans = span_exporter.get_finished_spans()

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

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate AI choice Event
    _choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
def test_invoke(instrument_legacy, span_exporter, log_exporter):
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

    spans = span_exporter.get_finished_spans()

    assert [
        "PromptTemplate.task",
        "ChatOpenAI.chat",
        "StrOutputParser.task",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_invoke_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    chat = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        streaming=True,
    )

    prompt_template = "write 10 lines of random text about ${product}"
    prompt = PromptTemplate.from_template(prompt_template)
    runnable = prompt | chat | StrOutputParser()
    response = runnable.invoke({"product": "colorful socks"})

    spans = span_exporter.get_finished_spans()

    assert [
        "PromptTemplate.task",
        "ChatOpenAI.chat",
        "StrOutputParser.task",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": prompt_template.format(product="colorful socks")},
    )

    # Validate AI choice Event
    _choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
def test_invoke_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()

    assert [
        "PromptTemplate.task",
        "ChatOpenAI.chat",
        "StrOutputParser.task",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate AI choice Event
    _choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
def test_stream(instrument_legacy, span_exporter, log_exporter):
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

    spans = span_exporter.get_finished_spans()

    assert [
        "PromptTemplate.task",
        "ChatOpenAI.chat",
        "StrOutputParser.task",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_stream_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    chat = ChatOpenAI(
        model="gpt-4",
        temperature=0,
    )

    prompt_template = "write 10 lines of random text about ${product}"
    prompt = PromptTemplate.from_template(prompt_template)
    runnable = prompt | chat | StrOutputParser()
    res = runnable.stream(
        input={"product": "colorful socks"},
        config={"configurable": {"session_id": 1234}},
    )
    chunks = [s for s in res]

    spans = span_exporter.get_finished_spans()

    assert [
        "PromptTemplate.task",
        "ChatOpenAI.chat",
        "StrOutputParser.task",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": prompt_template.format(product="colorful socks")},
    )

    # Validate AI choice Event
    _choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": "".join(chunks)},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
def test_stream_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()

    assert [
        "PromptTemplate.task",
        "ChatOpenAI.chat",
        "StrOutputParser.task",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate AI choice Event
    _choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_invoke(instrument_legacy, span_exporter, log_exporter):
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

    spans = span_exporter.get_finished_spans()

    assert [
        "PromptTemplate.task",
        "ChatOpenAI.chat",
        "StrOutputParser.task",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_invoke_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    chat = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        streaming=True,
    )

    prompt_template = "write 10 lines of random text about ${product}"
    prompt = PromptTemplate.from_template(prompt_template)
    runnable = prompt | chat | StrOutputParser()
    response = await runnable.ainvoke({"product": "colorful socks"})

    spans = span_exporter.get_finished_spans()

    assert [
        "PromptTemplate.task",
        "ChatOpenAI.chat",
        "StrOutputParser.task",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": prompt_template.format(product="colorful socks")},
    )

    # Validate AI choice Event
    _choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_invoke_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
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

    spans = span_exporter.get_finished_spans()

    assert [
        "PromptTemplate.task",
        "ChatOpenAI.chat",
        "StrOutputParser.task",
        "RunnableSequence.workflow",
    ] == [span.name for span in spans]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate AI choice Event
    _choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
def test_lcel_with_datetime(instrument_legacy, span_exporter, log_exporter):
    test_date = datetime.datetime(2023, 5, 17, 12, 34, 56)

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
    ).with_config(
        {
            "run_name": "DateTimeTestChain",
            "tags": ["datetime_test"],
            "metadata": {"timestamp": test_date, "test_name": "datetime_test"},
        }
    )

    chain.invoke({"input": "tell me a short joke"})

    spans = span_exporter.get_finished_spans()

    workflow_span = next(
        span for span in spans if span.name == "DateTimeTestChain.workflow"
    )

    entity_input = json.loads(
        workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT]
    )

    assert entity_input["metadata"]["timestamp"] == "2023-05-17T12:34:56"
    assert entity_input["metadata"]["test_name"] == "datetime_test"

    assert set(
        [
            "ChatPromptTemplate.task",
            "JsonOutputFunctionsParser.task",
            "ChatOpenAI.chat",
            "DateTimeTestChain.workflow",
        ]
    ) == set([span.name for span in spans])

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_lcel_with_datetime_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    test_date = datetime.datetime(2023, 5, 17, 12, 34, 56)

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
    ).with_config(
        {
            "run_name": "DateTimeTestChain",
            "tags": ["datetime_test"],
            "metadata": {"timestamp": test_date, "test_name": "datetime_test"},
        }
    )

    chain.invoke({"input": "tell me a short joke"})

    spans = span_exporter.get_finished_spans()

    assert set(
        [
            "ChatPromptTemplate.task",
            "JsonOutputFunctionsParser.task",
            "ChatOpenAI.chat",
            "DateTimeTestChain.workflow",
        ]
    ) == set([span.name for span in spans])

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
    _choice_event = {
        "index": 0,
        "finish_reason": "function_call",
        "message": {"content": ""},
        "tool_calls": [
            {
                "id": "",
                "function": {
                    "name": "Joke",
                    "arguments": '{"setup":"Why couldn\'t the bicycle stand up by '
                    'itself?","punchline":"Because it was two tired!"}',
                },
                "type": "function",
            }
        ],
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
def test_lcel_with_datetime_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    test_date = datetime.datetime(2023, 5, 17, 12, 34, 56)

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
    ).with_config(
        {
            "run_name": "DateTimeTestChain",
            "tags": ["datetime_test"],
            "metadata": {"timestamp": test_date, "test_name": "datetime_test"},
        }
    )

    chain.invoke({"input": "tell me a short joke"})

    spans = span_exporter.get_finished_spans()

    assert set(
        [
            "ChatPromptTemplate.task",
            "JsonOutputFunctionsParser.task",
            "ChatOpenAI.chat",
            "DateTimeTestChain.workflow",
        ]
    ) == set([span.name for span in spans])

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate system message Event
    assert_message_in_logs(logs[0], "gen_ai.system.message", {})

    # Validate user message Event
    assert_message_in_logs(logs[1], "gen_ai.user.message", {})

    # Validate AI choice Event
    _choice_event = {
        "index": 0,
        "finish_reason": "function_call",
        "message": {},
        "tool_calls": [{"function": {"name": "Joke"}, "id": "", "type": "function"}],
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", _choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "langchain"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
