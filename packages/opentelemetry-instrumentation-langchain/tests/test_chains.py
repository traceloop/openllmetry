import json

import pytest
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_cohere import ChatCohere
from langchain_openai import OpenAI
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_sequential_chain(instrument_legacy, span_exporter, log_exporter):
    llm = OpenAI(temperature=0.7)
    synopsis_template = """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

    Title: {title}
    Era: {era}
    Playwright: This is a synopsis for the above play:"""  # noqa: E501
    synopsis_prompt_template = PromptTemplate(
        input_variables=["title", "era"], template=synopsis_template
    )
    synopsis_chain = LLMChain(
        llm=llm, prompt=synopsis_prompt_template, output_key="synopsis", name="synopsis"
    )

    template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:"""  # noqa: E501
    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

    overall_chain = SequentialChain(
        chains=[synopsis_chain, review_chain],
        input_variables=["era", "title"],
        # Here we return multiple variables
        output_variables=["synopsis", "review"],
        verbose=True,
    )
    overall_chain.invoke(
        {"title": "Tragedy at sunset on the beach", "era": "Victorian England"}
    )

    spans = span_exporter.get_finished_spans()

    assert [
        "OpenAI.completion",
        "synopsis.task",
        "OpenAI.completion",
        "LLMChain.task",
        "SequentialChain.workflow",
    ] == [span.name for span in spans]

    workflow_span = next(
        span for span in spans if span.name == "SequentialChain.workflow"
    )
    task_spans = [
        span for span in spans if span.name in ["synopsis.task", "LLMChain.task"]
    ]
    llm_spans = [span for span in spans if span.name == "OpenAI.completion"]

    assert workflow_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "workflow"
    assert (
        workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME]
        == "SequentialChain"
    )
    assert all(
        span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "task"
        for span in task_spans
    )
    assert all(
        span.attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "SequentialChain"
        for span in spans
    )
    assert all(
        span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH]
        in ["synopsis", "LLMChain"]
        for span in llm_spans
    )

    synopsis_span = next(span for span in spans if span.name == "synopsis.task")
    review_span = next(span for span in spans if span.name == "LLMChain.task")

    data = json.loads(synopsis_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert data["inputs"] == {
        "title": "Tragedy at sunset on the beach",
        "era": "Victorian England",
    }
    assert data["kwargs"]["name"] == "synopsis"
    data = json.loads(synopsis_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert data["outputs"].keys() == {
        "synopsis",
    }

    data = json.loads(review_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert data["inputs"].keys() == {"title", "era", "synopsis"}
    assert data["kwargs"]["name"] == "LLMChain"
    data = json.loads(review_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert data["outputs"].keys() == {
        "review",
    }

    overall_span = next(
        span for span in spans if span.name == "SequentialChain.workflow"
    )
    data = json.loads(overall_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert data["inputs"] == {
        "title": "Tragedy at sunset on the beach",
        "era": "Victorian England",
    }
    assert data["kwargs"]["name"] == "SequentialChain"
    data = json.loads(overall_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert data["outputs"].keys() == {"synopsis", "review"}

    openai_span = next(span for span in spans if span.name == "OpenAI.completion")
    assert (
        openai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gpt-3.5-turbo-instruct"
    )
    assert (
        (openai_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL])
        == "gpt-3.5-turbo-instruct"
    )
    assert openai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_sequential_chain_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    llm = OpenAI(temperature=0.7)
    synopsis_template = """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

    Title: {title}
    Era: {era}
    Playwright: This is a synopsis for the above play:"""  # noqa: E501
    synopsis_prompt_template = PromptTemplate(
        input_variables=["title", "era"], template=synopsis_template
    )
    synopsis_chain = LLMChain(
        llm=llm, prompt=synopsis_prompt_template, output_key="synopsis", name="synopsis"
    )

    template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:"""  # noqa: E501
    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

    overall_chain = SequentialChain(
        chains=[synopsis_chain, review_chain],
        input_variables=["era", "title"],
        # Here we return multiple variables
        output_variables=["synopsis", "review"],
        verbose=True,
    )
    response = overall_chain.invoke(
        {"title": "Tragedy at sunset on the beach", "era": "Victorian England"}
    )

    spans = span_exporter.get_finished_spans()

    assert [
        "OpenAI.completion",
        "synopsis.task",
        "OpenAI.completion",
        "LLMChain.task",
        "SequentialChain.workflow",
    ] == [span.name for span in spans]

    workflow_span = next(
        span for span in spans if span.name == "SequentialChain.workflow"
    )
    task_spans = [
        span for span in spans if span.name in ["synopsis.task", "LLMChain.task"]
    ]
    llm_spans = [span for span in spans if span.name == "OpenAI.completion"]

    assert workflow_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "workflow"
    assert (
        workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME]
        == "SequentialChain"
    )
    assert all(
        span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "task"
        for span in task_spans
    )
    assert all(
        span.attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "SequentialChain"
        for span in spans
    )
    assert all(
        span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH]
        in ["synopsis", "LLMChain"]
        for span in llm_spans
    )

    openai_span = next(span for span in spans if span.name == "OpenAI.completion")
    assert (
        openai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gpt-3.5-turbo-instruct"
    )
    assert (
        (openai_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL])
        == "gpt-3.5-turbo-instruct"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    # Validate user message Event in the first chain
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {
            "content": synopsis_template.format(
                title="Tragedy at sunset on the beach", era="Victorian England"
            )
        },
    )

    # Validate AI choice Event in the first chain
    _choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response["synopsis"]},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)

    # Validate user message Event in the second chain
    assert_message_in_logs(
        logs[2],
        "gen_ai.user.message",
        {"content": template.format(synopsis=response["synopsis"])},
    )

    # Validate AI choice Event in the second chain
    _choice_event = {
        "index": 0,
        "finish_reason": "length",
        "message": {"content": response["review"]},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
def test_sequential_chain_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    llm = OpenAI(temperature=0.7)
    synopsis_template = """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

    Title: {title}
    Era: {era}
    Playwright: This is a synopsis for the above play:"""  # noqa: E501
    synopsis_prompt_template = PromptTemplate(
        input_variables=["title", "era"], template=synopsis_template
    )
    synopsis_chain = LLMChain(
        llm=llm, prompt=synopsis_prompt_template, output_key="synopsis", name="synopsis"
    )

    template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:"""  # noqa: E501
    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

    overall_chain = SequentialChain(
        chains=[synopsis_chain, review_chain],
        input_variables=["era", "title"],
        # Here we return multiple variables
        output_variables=["synopsis", "review"],
        verbose=True,
    )
    overall_chain.invoke(
        {"title": "Tragedy at sunset on the beach", "era": "Victorian England"}
    )

    spans = span_exporter.get_finished_spans()

    assert [
        "OpenAI.completion",
        "synopsis.task",
        "OpenAI.completion",
        "LLMChain.task",
        "SequentialChain.workflow",
    ] == [span.name for span in spans]

    workflow_span = next(
        span for span in spans if span.name == "SequentialChain.workflow"
    )
    task_spans = [
        span for span in spans if span.name in ["synopsis.task", "LLMChain.task"]
    ]
    llm_spans = [span for span in spans if span.name == "OpenAI.completion"]

    assert workflow_span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "workflow"
    assert (
        workflow_span.attributes[SpanAttributes.TRACELOOP_ENTITY_NAME]
        == "SequentialChain"
    )
    assert all(
        span.attributes[SpanAttributes.TRACELOOP_SPAN_KIND] == "task"
        for span in task_spans
    )
    assert all(
        span.attributes[SpanAttributes.TRACELOOP_WORKFLOW_NAME] == "SequentialChain"
        for span in spans
    )
    assert all(
        span.attributes[SpanAttributes.TRACELOOP_ENTITY_PATH]
        in ["synopsis", "LLMChain"]
        for span in llm_spans
    )

    openai_span = next(span for span in spans if span.name == "OpenAI.completion")
    assert (
        openai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gpt-3.5-turbo-instruct"
    )
    assert (
        (openai_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL])
        == "gpt-3.5-turbo-instruct"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    # Validate user message Event in the first chain
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate AI choice Event in the first chain
    _choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)

    # Validate user message Event in the second chain
    assert_message_in_logs(logs[2], "gen_ai.user.message", {})

    # Validate AI choice Event in the second chain
    _choice_event = {"index": 0, "finish_reason": "length", "message": {}}
    assert_message_in_logs(logs[3], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_asequential_chain(instrument_legacy, span_exporter, log_exporter):
    llm = OpenAI(temperature=0.7)
    synopsis_template = """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

    Title: {title}
    Era: {era}
    Playwright: This is a synopsis for the above play:"""  # noqa: E501
    synopsis_prompt_template = PromptTemplate(
        input_variables=["title", "era"], template=synopsis_template
    )
    synopsis_chain = LLMChain(
        llm=llm, prompt=synopsis_prompt_template, output_key="synopsis"
    )

    template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:"""  # noqa: E501
    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

    overall_chain = SequentialChain(
        chains=[synopsis_chain, review_chain],
        input_variables=["era", "title"],
        # Here we return multiple variables
        output_variables=["synopsis", "review"],
        verbose=True,
    )
    await overall_chain.ainvoke(
        {"title": "Tragedy at sunset on the beach", "era": "Victorian England"}
    )

    spans = span_exporter.get_finished_spans()

    assert [
        "OpenAI.completion",
        "LLMChain.task",
        "OpenAI.completion",
        "LLMChain.task",
        "SequentialChain.workflow",
    ] == [span.name for span in spans]

    synopsis_span, review_span = [
        span for span in spans if span.name == "LLMChain.task"
    ]

    data = json.loads(synopsis_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert data["inputs"] == {
        "title": "Tragedy at sunset on the beach",
        "era": "Victorian England",
    }
    assert data["kwargs"]["name"] == "LLMChain"
    data = json.loads(synopsis_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert data["outputs"].keys() == {
        "synopsis",
    }

    data = json.loads(review_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert data["inputs"].keys() == {"title", "era", "synopsis"}
    assert data["kwargs"]["name"] == "LLMChain"
    data = json.loads(review_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert data["outputs"].keys() == {
        "review",
    }

    overall_span = next(
        span for span in spans if span.name == "SequentialChain.workflow"
    )
    data = json.loads(overall_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert data["inputs"] == {
        "title": "Tragedy at sunset on the beach",
        "era": "Victorian England",
    }
    assert data["kwargs"]["name"] == "SequentialChain"
    data = json.loads(overall_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert data["outputs"].keys() == {"synopsis", "review"}

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_asequential_chain_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    llm = OpenAI(temperature=0.7)
    synopsis_template = """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

    Title: {title}
    Era: {era}
    Playwright: This is a synopsis for the above play:"""  # noqa: E501
    synopsis_prompt_template = PromptTemplate(
        input_variables=["title", "era"], template=synopsis_template
    )
    synopsis_chain = LLMChain(
        llm=llm, prompt=synopsis_prompt_template, output_key="synopsis"
    )

    template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:"""  # noqa: E501
    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

    overall_chain = SequentialChain(
        chains=[synopsis_chain, review_chain],
        input_variables=["era", "title"],
        # Here we return multiple variables
        output_variables=["synopsis", "review"],
        verbose=True,
    )
    response = await overall_chain.ainvoke(
        {"title": "Tragedy at sunset on the beach", "era": "Victorian England"}
    )

    spans = span_exporter.get_finished_spans()

    assert [
        "OpenAI.completion",
        "LLMChain.task",
        "OpenAI.completion",
        "LLMChain.task",
        "SequentialChain.workflow",
    ] == [span.name for span in spans]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    # Validate user message Event in the first chain
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {
            "content": synopsis_template.format(
                title="Tragedy at sunset on the beach", era="Victorian England"
            ),
        },
    )

    # Validate AI choice Event in the first chain
    _choice_event = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": response["synopsis"]},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)

    # Validate user message Event in the second chain
    assert_message_in_logs(
        logs[2],
        "gen_ai.user.message",
        {"content": template.format(synopsis=response["synopsis"])},
    )

    # Validate AI choice Event in the second chain
    _choice_event = {
        "index": 0,
        "finish_reason": "length",
        "message": {"content": response["review"]},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_asequential_chain_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    llm = OpenAI(temperature=0.7)
    synopsis_template = """You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

    Title: {title}
    Era: {era}
    Playwright: This is a synopsis for the above play:"""  # noqa: E501
    synopsis_prompt_template = PromptTemplate(
        input_variables=["title", "era"], template=synopsis_template
    )
    synopsis_chain = LLMChain(
        llm=llm, prompt=synopsis_prompt_template, output_key="synopsis"
    )

    template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:"""  # noqa: E501
    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

    overall_chain = SequentialChain(
        chains=[synopsis_chain, review_chain],
        input_variables=["era", "title"],
        # Here we return multiple variables
        output_variables=["synopsis", "review"],
        verbose=True,
    )
    await overall_chain.ainvoke(
        {"title": "Tragedy at sunset on the beach", "era": "Victorian England"}
    )

    spans = span_exporter.get_finished_spans()

    assert [
        "OpenAI.completion",
        "LLMChain.task",
        "OpenAI.completion",
        "LLMChain.task",
        "SequentialChain.workflow",
    ] == [span.name for span in spans]

    synopsis_span, review_span = [
        span for span in spans if span.name == "LLMChain.task"
    ]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    # Validate user message Event in the first chain
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate AI choice Event in the first chain
    _choice_event = {"index": 0, "finish_reason": "stop", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)

    # Validate user message Event in the second chain
    assert_message_in_logs(logs[2], "gen_ai.user.message", {})

    # Validate AI choice Event in the second chain
    _choice_event = {"index": 0, "finish_reason": "length", "message": {}}
    assert_message_in_logs(logs[3], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
def test_stream(instrument_legacy, span_exporter, log_exporter):
    chat = ChatCohere(model="command", temperature=0.75)
    prompt = PromptTemplate.from_template(
        "write 2 lines of random text about ${product}"
    )
    runnable = prompt | chat | StrOutputParser()

    chunks = list(runnable.stream({"product": "colorful socks"}))
    spans = span_exporter.get_finished_spans()

    assert set(
        [
            "PromptTemplate.task",
            "StrOutputParser.task",
            "ChatCohere.chat",
            "RunnableSequence.workflow",
        ]
    ) == set([span.name for span in spans])
    assert len(chunks) == 62

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_stream_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    chat = ChatCohere(model="command", temperature=0.75)
    prompt_template = "write 2 lines of random text about ${product}"
    prompt = PromptTemplate.from_template(prompt_template)
    runnable = prompt | chat | StrOutputParser()

    chunks = list(runnable.stream({"product": "colorful socks"}))
    spans = span_exporter.get_finished_spans()

    assert set(
        [
            "PromptTemplate.task",
            "StrOutputParser.task",
            "ChatCohere.chat",
            "RunnableSequence.workflow",
        ]
    ) == set([span.name for span in spans])
    assert len(chunks) == 62

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {
            "content": prompt_template.format(product="colorful socks"),
        },
    )

    # Validate AI choice Event
    _choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": "".join(chunks)},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
def test_stream_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    chat = ChatCohere(model="command", temperature=0.75)
    prompt = PromptTemplate.from_template(
        "write 2 lines of random text about ${product}"
    )
    runnable = prompt | chat | StrOutputParser()

    chunks = list(runnable.stream({"product": "colorful socks"}))
    spans = span_exporter.get_finished_spans()

    assert set(
        [
            "PromptTemplate.task",
            "StrOutputParser.task",
            "ChatCohere.chat",
            "RunnableSequence.workflow",
        ]
    ) == set([span.name for span in spans])
    assert len(chunks) == 62

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate AI choice Event
    _choice_event = {"index": 0, "finish_reason": "unknown", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_astream(instrument_legacy, span_exporter, log_exporter):
    chat = ChatCohere(model="command", temperature=0.75)
    prompt = PromptTemplate.from_template(
        "write 2 lines of random text about ${product}"
    )
    runnable = prompt | chat | StrOutputParser()

    chunks = []
    async for chunk in runnable.astream({"product": "colorful socks"}):
        chunks.append(chunk)
    spans = span_exporter.get_finished_spans()

    assert set(
        [
            "PromptTemplate.task",
            "ChatCohere.chat",
            "StrOutputParser.task",
            "RunnableSequence.workflow",
        ]
    ) == set([span.name for span in spans])
    assert len(chunks) == 144

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_astream_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    chat = ChatCohere(model="command", temperature=0.75)
    prompt_template = "write 2 lines of random text about ${product}"
    prompt = PromptTemplate.from_template(prompt_template)
    runnable = prompt | chat | StrOutputParser()

    chunks = []
    async for chunk in runnable.astream({"product": "colorful socks"}):
        chunks.append(chunk)
    spans = span_exporter.get_finished_spans()

    assert set(
        [
            "PromptTemplate.task",
            "ChatCohere.chat",
            "StrOutputParser.task",
            "RunnableSequence.workflow",
        ]
    ) == set([span.name for span in spans])
    assert len(chunks) == 144

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": prompt_template.format(product="colorful socks")},
    )

    # Validate AI choice Event
    # _choice_event = {
    #     "index": 0,
    #     "finish_reason": "unknown",
    #     "message": {"content": "".join(chunks)},
    # }
    # assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_astream_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    chat = ChatCohere(model="command", temperature=0.75)
    prompt = PromptTemplate.from_template(
        "write 2 lines of random text about ${product}"
    )
    runnable = prompt | chat | StrOutputParser()

    chunks = []
    async for chunk in runnable.astream({"product": "colorful socks"}):
        chunks.append(chunk)
    spans = span_exporter.get_finished_spans()

    assert set(
        [
            "PromptTemplate.task",
            "ChatCohere.chat",
            "StrOutputParser.task",
            "RunnableSequence.workflow",
        ]
    ) == set([span.name for span in spans])
    assert len(chunks) == 144

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate AI choice Event
    # _choice_event = {"index": 0, "finish_reason": "unknown", "message": {}}
    # assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "langchain"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
