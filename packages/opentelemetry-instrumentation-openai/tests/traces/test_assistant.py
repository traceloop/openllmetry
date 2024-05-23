import time
from typing_extensions import override

import pytest
from openai import AssistantEventHandler
from opentelemetry.semconv.ai import SpanAttributes


@pytest.fixture
def assistant(openai_client):
    return openai_client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4-turbo-preview",
    )


@pytest.mark.vcr
def test_new_assistant(exporter, openai_client, assistant):
    thread = openai_client.beta.threads.create()

    openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="I need to solve the equation `3x + 11 = 14`. Can you help me?",
    )

    run = openai_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Jane Doe. The user has a premium account.",
    )

    while run.status in ["queued", "in_progress", "cancelling"]:
        time.sleep(1)  # Wait for 1 second
        run = openai_client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id
        )

    messages = openai_client.beta.threads.messages.list(
        thread_id=thread.id, order="asc"
    )
    spans = exporter.get_finished_spans()

    assert run.status == "completed"

    assert [span.name for span in spans] == [
        "openai.assistant.run",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4-turbo-preview"
    assert open_ai_span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-4-turbo-preview"
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "You are a personal math tutor. Write and run code to answer math questions."
    )
    assert open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert (
        open_ai_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.1.content")
        == "Please address the user as Jane Doe. The user has a premium account."
    )
    assert open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "system"

    for idx, message in enumerate(messages.data):
        assert (
            open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.content"]
            == message.content[0].text.value
        )
        assert open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.role"] == message.role


@pytest.mark.vcr
def test_existing_assistant(exporter, openai_client):
    thread = openai_client.beta.threads.create()

    openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="I need to solve the equation `3x + 11 = 14`. Can you help me?",
    )

    run = openai_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id="asst_rr3RGZE5iqoMCxqFOpb7AZmr",
        instructions="Please address the user as Jane Doe. The user has a premium account.",
    )

    while run.status in ["queued", "in_progress", "cancelling"]:
        time.sleep(1)  # Wait for 1 second
        run = openai_client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id
        )

    messages = openai_client.beta.threads.messages.list(
        thread_id=thread.id, order="asc"
    )
    spans = exporter.get_finished_spans()

    assert run.status == "completed"

    assert [span.name for span in spans] == [
        "openai.assistant.run",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4-turbo-preview"
    assert open_ai_span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-4-turbo-preview"
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "You are a personal math tutor. Write and run code to answer math questions."
    )
    assert open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert (
        open_ai_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.1.content")
        == "Please address the user as Jane Doe. The user has a premium account."
    )
    assert open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "system"

    for idx, message in enumerate(messages.data):
        assert (
            open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.content"]
            == message.content[0].text.value
        )
        assert open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.role"] == message.role


@pytest.mark.vcr
def test_streaming_new_assistant(exporter, openai_client, assistant):
    thread = openai_client.beta.threads.create()

    openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="I need to solve the equation `3x + 11 = 14`. Can you help me?",
    )

    assistant_messages = []

    class EventHandler(AssistantEventHandler):
        @override
        def on_text_created(self, text) -> None:
            assistant_messages.append("")

        @override
        def on_text_delta(self, delta, snapshot):
            assistant_messages[-1] += delta.value

    with openai_client.beta.threads.runs.create_and_stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Jane Doe. The user has a premium account.",
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.assistant.run_stream",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4-turbo-preview"
    assert open_ai_span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-4-turbo-preview"
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "You are a personal math tutor. Write and run code to answer math questions."
    )
    assert open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert (
        open_ai_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.1.content")
        == "Please address the user as Jane Doe. The user has a premium account."
    )
    assert open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "system"

    for idx, message in enumerate(assistant_messages):
        assert open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.content"] == message
        assert open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.role"] == "assistant"


@pytest.mark.vcr
def test_streaming_existing_assistant(exporter, openai_client):
    thread = openai_client.beta.threads.create()

    openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="I need to solve the equation `3x + 11 = 14`. Can you help me?",
    )

    assistant_messages = []

    class EventHandler(AssistantEventHandler):
        @override
        def on_text_created(self, text) -> None:
            assistant_messages.append("")

        @override
        def on_text_delta(self, delta, snapshot):
            assistant_messages[-1] += delta.value

    with openai_client.beta.threads.runs.create_and_stream(
        thread_id=thread.id,
        assistant_id="asst_rr3RGZE5iqoMCxqFOpb7AZmr",
        instructions="Please address the user as Jane Doe. The user has a premium account.",
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.assistant.run_stream",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_MODEL] == "gpt-4-turbo-preview"
    assert open_ai_span.attributes[SpanAttributes.LLM_RESPONSE_MODEL] == "gpt-4-turbo-preview"
    assert (
        open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.content"]
        == "You are a personal math tutor. Write and run code to answer math questions."
    )
    assert open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.0.role"] == "system"
    assert (
        open_ai_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.1.content")
        == "Please address the user as Jane Doe. The user has a premium account."
    )
    assert open_ai_span.attributes[f"{SpanAttributes.LLM_PROMPTS}.1.role"] == "system"

    for idx, message in enumerate(assistant_messages):
        assert open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.content"] == message
        assert open_ai_span.attributes[f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.role"] == "assistant"
