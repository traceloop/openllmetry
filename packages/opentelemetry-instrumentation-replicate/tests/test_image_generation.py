import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


@pytest.mark.vcr
def test_replicate_image_generation_legacy(
    instrument_legacy, replicate_client, span_exporter, log_exporter
):
    replicate_client.run(
        "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
        input={"prompt": "robots"},
    )
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "replicate.run",
    ]

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_replicate_image_generation_with_events_with_content(
    instrument_with_content, replicate_client, span_exporter, log_exporter
):
    response = replicate_client.run(
        "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
        input={"prompt": "robots"},
    )
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "replicate.run",
    ]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", {"content": "robots"}
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response[0]},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_replicate_image_generation_with_events_with_no_content(
    instrument_with_no_content, replicate_client, span_exporter, log_exporter
):
    replicate_client.run(
        "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
        input={"prompt": "robots"},
    )
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "replicate.run",
    ]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_replicate_image_generation_predictions_legacy(
    instrument_legacy, replicate_client, span_exporter, log_exporter
):
    model = replicate_client.models.get("kvfrans/clipdraw")
    version = model.versions.get(
        "5797a99edc939ea0e9242d5e8c9cb3bc7d125b1eac21bda852e5cb79ede2cd9b"
    )
    replicate_client.predictions.create(version, input={"prompt": "robots"})

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "replicate.predictions.create",
    ]

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_replicate_image_generation_predictions_with_events_with_content(
    instrument_with_content, replicate_client, span_exporter, log_exporter
):
    model = replicate_client.models.get("kvfrans/clipdraw")
    version = model.versions.get(
        "5797a99edc939ea0e9242d5e8c9cb3bc7d125b1eac21bda852e5cb79ede2cd9b"
    )
    response = replicate_client.predictions.create(version, input={"prompt": "robots"})

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "replicate.predictions.create",
    ]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(
        user_message_log, "gen_ai.user.message", {"content": "robots"}
    )

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response.output},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_replicate_image_generation_predictions_with_events_with_no_content(
    instrument_with_no_content, replicate_client, span_exporter, log_exporter
):
    model = replicate_client.models.get("kvfrans/clipdraw")
    version = model.versions.get(
        "5797a99edc939ea0e9242d5e8c9cb3bc7d125b1eac21bda852e5cb79ede2cd9b"
    )
    replicate_client.predictions.create(version, input={"prompt": "robots"})

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "replicate.predictions.create",
    ]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    user_message_log = logs[0]
    assert_message_in_logs(user_message_log, "gen_ai.user.message", {})

    # Validate the ai response
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "replicate"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
