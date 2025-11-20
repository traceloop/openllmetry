import pytest
from openai import AssistantEventHandler
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes
from typing_extensions import override


@pytest.fixture
def assistant(openai_client):
    return openai_client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-4-turbo-preview",
    )


@pytest.mark.vcr
def test_new_assistant(
    instrument_legacy, span_exporter, log_exporter, openai_client, assistant
):
    thread = openai_client.beta.threads.create()
    user_message = "I need to solve the equation `3x + 11 = 14`. Can you help me?"

    openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message,
    )

    run = openai_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Jane Doe. The user has a premium account.",
    )

    while run.status in ["queued", "in_progress", "cancelling"]:
        # in the original run, it waits for 1 second
        # Now that we have VCR cassetes recorded, we don't need to wait
        # time.sleep(1)
        run = openai_client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id
        )

    messages = openai_client.beta.threads.messages.list(
        thread_id=thread.id, order="asc"
    )
    spans = span_exporter.get_finished_spans()

    assert run.status == "completed"

    assert [span.name for span in spans] == [
        "openai.assistant.run",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "You are a personal math tutor. Write and run code to answer math questions."
    )
    assert open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "system"
    assert (
        open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.1.content")
        == "Please address the user as Jane Doe. The user has a premium account."
    )
    assert open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"] == "system"
    assert open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.role"] == "user"
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.2.content"]
        == user_message
    )
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 155
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 145
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"

    completion_index = 0
    for message in messages.data:
        if message.role in ["user", "system"]:
            continue
        assert (
            open_ai_span.attributes[
                f"{GenAIAttributes.GEN_AI_COMPLETION}.{completion_index}.content"
            ]
            == message.content[0].text.value
        )
        assert (
            open_ai_span.attributes[
                f"{GenAIAttributes.GEN_AI_COMPLETION}.{completion_index}.role"
            ]
            == message.role
        )
        assert (
            open_ai_span.attributes[f"gen_ai.response.{completion_index}.id"]
            == message.id
        )
        completion_index += 1

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_new_assistant_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client, assistant
):
    thread = openai_client.beta.threads.create()
    user_message = "I need to solve the equation `3x + 11 = 14`. Can you help me?"

    openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message,
    )

    model_instructions = (
        "Please address the user as Jane Doe. The user has a premium account."
    )
    run = openai_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions=model_instructions,
    )

    while run.status in ["queued", "in_progress", "cancelling"]:
        # in the original run, it waits for 1 second
        # Now that we have VCR cassetes recorded, we don't need to wait
        # time.sleep(1)
        run = openai_client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id
        )

    messages = openai_client.beta.threads.messages.list(
        thread_id=thread.id, order="asc"
    )
    spans = span_exporter.get_finished_spans()

    assert run.status == "completed"

    assert [span.name for span in spans] == [
        "openai.assistant.run",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == "gpt-4-turbo-preview"
    )

    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 155
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 145
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    # Validate run system message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.system.message",
        {
            "content": "You are a personal math tutor. Write and run code to answer math questions.",
        },
    )

    # Validate assistent system message Event
    assert_message_in_logs(
        logs[1], "gen_ai.system.message", {"content": model_instructions}
    )

    # Validate user message Event
    assert_message_in_logs(logs[2], "gen_ai.user.message", {"content": user_message})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {
            "content": messages.data[-1].content[0].text.value,
        },
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_new_assistant_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client, assistant
):
    thread = openai_client.beta.threads.create()
    user_message = "I need to solve the equation `3x + 11 = 14`. Can you help me?"

    openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message,
    )

    run = openai_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Jane Doe. The user has a premium account.",
    )

    while run.status in ["queued", "in_progress", "cancelling"]:
        # in the original run, it waits for 1 second
        # Now that we have VCR cassetes recorded, we don't need to wait
        # time.sleep(1)
        run = openai_client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id
        )

    openai_client.beta.threads.messages.list(thread_id=thread.id, order="asc")
    spans = span_exporter.get_finished_spans()

    assert run.status == "completed"

    assert [span.name for span in spans] == [
        "openai.assistant.run",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 155
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 145
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    # Validate run system message Event
    assert_message_in_logs(logs[0], "gen_ai.system.message", {})

    # Validate assistent system message Event
    assert_message_in_logs(logs[1], "gen_ai.system.message", {})

    # Validate user message Event
    user_message_log = logs[2]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "unknown", "message": {}}
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_new_assistant_with_polling(
    instrument_legacy, span_exporter, log_exporter, openai_client, assistant
):
    thread = openai_client.beta.threads.create()
    user_message = "I need to solve the equation `3x + 11 = 14`. Can you help me?"

    openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message,
    )

    run = openai_client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Jane Doe. The user has a premium account.",
        # Original test waited for much longer, but now that we have VCR cassetes recorded,
        # we don't need to wait
        poll_interval_ms=20,
    )

    messages = openai_client.beta.threads.messages.list(
        thread_id=thread.id, order="asc"
    )
    spans = span_exporter.get_finished_spans()

    assert run.status == "completed"

    assert [span.name for span in spans] == [
        "openai.assistant.run",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes["llm.request.type"] == "chat"
    assert open_ai_span.attributes["gen_ai.request.model"] == "gpt-4-turbo-preview"
    assert open_ai_span.attributes["gen_ai.response.model"] == "gpt-4-turbo-preview"
    assert (
        open_ai_span.attributes["gen_ai.prompt.0.content"]
        == "You are a personal math tutor. Write and run code to answer math questions."
    )
    assert open_ai_span.attributes["gen_ai.prompt.0.role"] == "system"
    assert (
        open_ai_span.attributes.get("gen_ai.prompt.1.content")
        == "Please address the user as Jane Doe. The user has a premium account."
    )
    assert open_ai_span.attributes["gen_ai.prompt.1.role"] == "system"
    assert open_ai_span.attributes["gen_ai.prompt.2.role"] == "user"
    assert open_ai_span.attributes["gen_ai.prompt.2.content"] == user_message
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 86
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 374
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"

    completion_index = 0
    for message in messages.data:
        if message.role in ["user", "system"]:
            continue
        assert (
            open_ai_span.attributes[
                f"{GenAIAttributes.GEN_AI_COMPLETION}.{completion_index}.content"
            ]
            == message.content[0].text.value
        )
        assert (
            open_ai_span.attributes[
                f"{GenAIAttributes.GEN_AI_COMPLETION}.{completion_index}.role"
            ]
            == message.role
        )
        assert (
            open_ai_span.attributes[f"gen_ai.response.{completion_index}.id"]
            == message.id
        )
        completion_index += 1

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_new_assistant_with_polling_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client, assistant
):
    thread = openai_client.beta.threads.create()
    user_message = "I need to solve the equation `3x + 11 = 14`. Can you help me?"

    openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message,
    )

    run = openai_client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Jane Doe. The user has a premium account.",
        # Original test waited for much longer, but now that we have VCR cassetes recorded,
        # we don't need to wait
        poll_interval_ms=20,
    )

    messages = openai_client.beta.threads.messages.list(
        thread_id=thread.id, order="asc"
    )
    spans = span_exporter.get_finished_spans()

    assert run.status == "completed"

    assert [span.name for span in spans] == [
        "openai.assistant.run",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes["llm.request.type"] == "chat"
    assert open_ai_span.attributes["gen_ai.request.model"] == "gpt-4-turbo-preview"
    assert open_ai_span.attributes["gen_ai.response.model"] == "gpt-4-turbo-preview"
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 86
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 374
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    # Validate run system message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.system.message",
        {
            "content": "You are a personal math tutor. Write and run code to answer math questions."
        },
    )

    # Validate assistent system message Event
    assert_message_in_logs(
        logs[1],
        "gen_ai.system.message",
        {
            "content": "Please address the user as Jane Doe. The user has a premium account.",
        },
    )

    # Validate user message Event
    user_message_log = logs[2]
    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", {"content": user_message}
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": messages.data[-1].content[0].text.value},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_new_assistant_with_polling_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client, assistant
):
    thread = openai_client.beta.threads.create()
    user_message = "I need to solve the equation `3x + 11 = 14`. Can you help me?"

    openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message,
    )

    run = openai_client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Jane Doe. The user has a premium account.",
        # Original test waited for much longer, but now that we have VCR cassetes recorded,
        # we don't need to wait
        poll_interval_ms=20,
    )

    openai_client.beta.threads.messages.list(thread_id=thread.id, order="asc")
    spans = span_exporter.get_finished_spans()

    assert run.status == "completed"

    assert [span.name for span in spans] == [
        "openai.assistant.run",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes["llm.request.type"] == "chat"
    assert open_ai_span.attributes["gen_ai.request.model"] == "gpt-4-turbo-preview"
    assert open_ai_span.attributes["gen_ai.response.model"] == "gpt-4-turbo-preview"
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 86
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 374
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    # Validate run system message Event
    assert_message_in_logs(logs[0], "gen_ai.system.message", {})

    # Validate assistent system message Event
    assert_message_in_logs(logs[1], "gen_ai.system.message", {})

    # Validate user message Event
    user_message_log = logs[2]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "unknown", "message": {}}
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_existing_assistant(
    instrument_legacy, span_exporter, log_exporter, openai_client
):
    thread = openai_client.beta.threads.create()
    user_message = "I need to solve the equation `3x + 11 = 14`. Can you help me?"

    openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message,
    )

    run = openai_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id="asst_rr3RGZE5iqoMCxqFOpb7AZmr",
        instructions="Please address the user as Jane Doe. The user has a premium account.",
    )

    while run.status in ["queued", "in_progress", "cancelling"]:
        # in the original run, it waits for 1 second
        # Now that we have VCR cassetes recorded, we don't need to wait
        # time.sleep(1)
        run = openai_client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id
        )

    messages = openai_client.beta.threads.messages.list(
        thread_id=thread.id, order="asc"
    )
    spans = span_exporter.get_finished_spans()

    assert run.status == "completed"

    assert [span.name for span in spans] == [
        "openai.assistant.run",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "You are a personal math tutor. Write and run code to answer math questions."
    )
    assert open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "system"
    assert (
        open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.1.content")
        == "Please address the user as Jane Doe. The user has a premium account."
    )
    assert open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"] == "system"
    assert open_ai_span.attributes["gen_ai.prompt.2.role"] == "user"
    assert open_ai_span.attributes["gen_ai.prompt.2.content"] == user_message
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 170
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 639
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"

    completion_index = 0
    for message in messages.data:
        if message.role in ["user", "system"]:
            continue
        assert (
            open_ai_span.attributes[
                f"{GenAIAttributes.GEN_AI_COMPLETION}.{completion_index}.content"
            ]
            == message.content[0].text.value
        )
        assert (
            open_ai_span.attributes[
                f"{GenAIAttributes.GEN_AI_COMPLETION}.{completion_index}.role"
            ]
            == message.role
        )
        assert (
            open_ai_span.attributes[f"gen_ai.response.{completion_index}.id"]
            == message.id
        )
        completion_index += 1

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_existing_assistant_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
    thread = openai_client.beta.threads.create()
    user_message = "I need to solve the equation `3x + 11 = 14`. Can you help me?"

    openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message,
    )

    run = openai_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id="asst_rr3RGZE5iqoMCxqFOpb7AZmr",
        instructions="Please address the user as Jane Doe. The user has a premium account.",
    )

    while run.status in ["queued", "in_progress", "cancelling"]:
        # in the original run, it waits for 1 second
        # Now that we have VCR cassetes recorded, we don't need to wait
        # time.sleep(1)
        run = openai_client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id
        )

    messages = openai_client.beta.threads.messages.list(
        thread_id=thread.id, order="asc"
    )
    spans = span_exporter.get_finished_spans()

    assert run.status == "completed"

    assert [span.name for span in spans] == [
        "openai.assistant.run",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == "gpt-4-turbo-preview"
    )

    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 170
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 639
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 5

    # Validate run system message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.system.message",
        {
            "content": "You are a personal math tutor. Write and run code to answer math questions.",
        },
    )

    # Validate assistent system message Event
    assert_message_in_logs(
        logs[1],
        "gen_ai.system.message",
        {
            "content": "Please address the user as Jane Doe. The user has a premium account.",
        },
    )

    # Validate user message Event
    user_message_log = logs[2]
    assert_message_in_logs(
        user_message_log,
        "gen_ai.user.message",
        {"content": user_message},
    )

    # Validate the first ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": messages.data[-2].content[0].text.value},
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)

    # Validate the second ai response
    choice_event = {
        "index": 1,
        "finish_reason": "unknown",
        "message": {"content": messages.data[-1].content[0].text.value},
    }
    assert_message_in_logs(logs[4], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_existing_assistant_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client
):
    thread = openai_client.beta.threads.create()
    user_message = "I need to solve the equation `3x + 11 = 14`. Can you help me?"

    openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message,
    )

    run = openai_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id="asst_rr3RGZE5iqoMCxqFOpb7AZmr",
        instructions="Please address the user as Jane Doe. The user has a premium account.",
    )

    while run.status in ["queued", "in_progress", "cancelling"]:
        # in the original run, it waits for 1 second
        # Now that we have VCR cassetes recorded, we don't need to wait
        # time.sleep(1)
        run = openai_client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id
        )

    openai_client.beta.threads.messages.list(thread_id=thread.id, order="asc")
    spans = span_exporter.get_finished_spans()

    assert run.status == "completed"

    assert [span.name for span in spans] == [
        "openai.assistant.run",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == "gpt-4-turbo-preview"
    )

    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 170
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 639
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 5

    # Validate run system message Event
    assert_message_in_logs(logs[0], "gen_ai.system.message", {})

    # Validate assistent system message Event
    assert_message_in_logs(logs[1], "gen_ai.system.message", {})

    # Validate user message Event
    user_message_log = logs[2]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the first ai response
    choice_event = {"index": 0, "finish_reason": "unknown", "message": {}}
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)

    # Validate the second ai response
    choice_event = {"index": 1, "finish_reason": "unknown", "message": {}}
    assert_message_in_logs(logs[4], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_streaming_new_assistant(
    instrument_legacy, span_exporter, log_exporter, openai_client, assistant
):
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

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.assistant.run_stream",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "You are a personal math tutor. Write and run code to answer math questions."
    )
    assert open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "system"
    assert (
        open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.1.content")
        == "Please address the user as Jane Doe. The user has a premium account."
    )
    assert open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"] == "system"

    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 790
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 225
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"

    for idx, message in enumerate(assistant_messages):
        assert (
            open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.{idx}.content"]
            == message
        )
        assert (
            open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.{idx}.role"]
            == "assistant"
        )
        assert open_ai_span.attributes[f"gen_ai.response.{idx}.id"].startswith("msg")

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_streaming_new_assistant_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client, assistant
):
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

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.assistant.run_stream",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == "gpt-4-turbo-preview"
    )

    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 790
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 225
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    # Validate run system message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.system.message",
        {
            "content": "You are a personal math tutor. Write and run code to answer math questions.",
        },
    )

    # Validate assistent system message Event
    assert_message_in_logs(
        logs[1],
        "gen_ai.system.message",
        {
            "content": "Please address the user as Jane Doe. The user has a premium account.",
        },
    )

    # Validate the first ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {
            "content": [
                {
                    "text": {"annotations": [], "value": assistant_messages[0]},
                    "type": "text",
                }
            ],
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)

    # Validate the second ai response
    choice_event = {
        "index": 1,
        "finish_reason": "unknown",
        "message": {
            "content": [
                {
                    "text": {"annotations": [], "value": assistant_messages[1]},
                    "type": "text",
                }
            ],
        },
    }
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_streaming_new_assistant_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client, assistant
):
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

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.assistant.run_stream",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == "gpt-4-turbo-preview"
    )

    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 790
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 225
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 4

    # Validate run system message Event
    assert_message_in_logs(logs[0], "gen_ai.system.message", {})

    # Validate assistent system message Event
    assert_message_in_logs(logs[1], "gen_ai.system.message", {})

    # Validate the first ai response
    choice_event = {"index": 0, "finish_reason": "unknown", "message": {}}
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)

    # Validate the second ai response
    choice_event = {"index": 1, "finish_reason": "unknown", "message": {}}
    assert_message_in_logs(logs[3], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_streaming_existing_assistant(
    instrument_legacy, span_exporter, log_exporter, openai_client
):
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

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.assistant.run_stream",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert (
        open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.content"]
        == "You are a personal math tutor. Write and run code to answer math questions."
    )
    assert open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.0.role"] == "system"
    assert (
        open_ai_span.attributes.get(f"{GenAIAttributes.GEN_AI_PROMPT}.1.content")
        == "Please address the user as Jane Doe. The user has a premium account."
    )
    assert open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_PROMPT}.1.role"] == "system"
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 364
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 88
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"

    for idx, message in enumerate(assistant_messages):
        assert (
            open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.{idx}.content"]
            == message
        )
        assert (
            open_ai_span.attributes[f"{GenAIAttributes.GEN_AI_COMPLETION}.{idx}.role"]
            == "assistant"
        )
        assert open_ai_span.attributes[f"gen_ai.response.{idx}.id"].startswith("msg_")

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_streaming_existing_assistant_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter, openai_client
):
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

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.assistant.run_stream",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == "gpt-4-turbo-preview"
    )

    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 364
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 88
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate run system message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.system.message",
        {
            "content": "You are a personal math tutor. Write and run code to answer math questions.",
        },
    )

    # Validate assistent system message Event
    assert_message_in_logs(
        logs[1],
        "gen_ai.system.message",
        {
            "content": "Please address the user as Jane Doe. The user has a premium account.",
        },
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {
            "content": [
                {
                    "text": {"annotations": [], "value": assistant_messages[0]},
                    "type": "text",
                }
            ],
        },
    }
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_streaming_existing_assistant_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter, openai_client
):
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

    spans = span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "openai.assistant.run_stream",
    ]
    open_ai_span = spans[0]
    assert open_ai_span.attributes[SpanAttributes.LLM_REQUEST_TYPE] == "chat"
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "gpt-4-turbo-preview"
    )
    assert (
        open_ai_span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == "gpt-4-turbo-preview"
    )

    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 364
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 88
    assert open_ai_span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 3

    # Validate run system message Event
    assert_message_in_logs(logs[0], "gen_ai.system.message", {})

    # Validate assistent system message Event
    assert_message_in_logs(logs[1], "gen_ai.system.message", {})

    # Validate the ai response
    choice_event = {"index": 0, "finish_reason": "unknown", "message": {}}
    assert_message_in_logs(logs[2], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.OPENAI.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
