import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    event_attributes as EventAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes


@pytest.mark.vcr
def test_cohere_rerank_legacy(
    span_exporter, log_exporter, instrument_legacy, cohere_client
):
    query = "What is the capital of the United States?"
    documents = [
        "Carson City is the capital city of the American state of Nevada."
        + " At the  2010 United States Census, Carson City had a population of 55,274.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands"
        + " in the Pacific Ocean that are a political division controlled by the "
        + "United States. Its capital is Saipan.",
        "Charlotte Amalie is the capital and largest city of the United States "
        + "Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
        "Washington, D.C. (also known as simply Washington or D.C., and officially "
        + "as the District of Columbia) is the capital of the United States. It is a federal district. ",
        "Capital punishment (the death penalty) has existed in the United States "
        + "since before the United States was a country. As of 2017, capital "
        + "punishment is legal in 30 of the 50 states.",
        "North Dakota is a state in the United States. 672,591 people lived"
        + " in North Dakota in the year 2010. The capital and seat of government is Bismarck.",
    ]

    response = cohere_client.rerank(
        query=query, documents=documents, top_n=3, model="rerank-multilingual-v2.0"
    )

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.rerank"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "rerank"
    assert (
        cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL)
        == "rerank-multilingual-v2.0"
    )
    assert (
        cohere_span.attributes.get(
            f"{SpanAttributes.LLM_PROMPTS}.{len(documents)}.role"
        )
        == "user"
    )
    assert (
        cohere_span.attributes.get(
            f"{SpanAttributes.LLM_PROMPTS}.{len(documents)}.content"
        )
        == query
    )

    for i, doc in enumerate(documents):
        assert (
            cohere_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.{i}.role")
            == "system"
        )
        assert (
            cohere_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.{i}.content")
            == doc
        )

    for idx, result in enumerate(response.results):
        assert (
            cohere_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.role")
            == "assistant"
        )
        assert (
            cohere_span.attributes.get(
                f"{SpanAttributes.LLM_COMPLETIONS}.{idx}.content"
            )
            == f"Doc {result.index}, Score: {result.relevance_score}"
        )
    assert (
        cohere_span.attributes.get("gen_ai.response.id")
        == "c995b490-c717-411c-8fd0-8bf993cd8382"
    )

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_cohere_rerank_with_events_with_content(
    span_exporter, log_exporter, instrument_with_content, cohere_client
):
    query = "What is the capital of the United States?"
    documents = [
        "Carson City is the capital city of the American state of Nevada."
        + " At the  2010 United States Census, Carson City had a population of 55,274.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands"
        + " in the Pacific Ocean that are a political division controlled by the "
        + "United States. Its capital is Saipan.",
        "Charlotte Amalie is the capital and largest city of the United States "
        + "Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
        "Washington, D.C. (also known as simply Washington or D.C., and officially "
        + "as the District of Columbia) is the capital of the United States. It is a federal district. ",
        "Capital punishment (the death penalty) has existed in the United States "
        + "since before the United States was a country. As of 2017, capital "
        + "punishment is legal in 30 of the 50 states.",
        "North Dakota is a state in the United States. 672,591 people lived"
        + " in North Dakota in the year 2010. The capital and seat of government is Bismarck.",
    ]

    response = cohere_client.rerank(
        query=query, documents=documents, top_n=3, model="rerank-multilingual-v2.0"
    )

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.rerank"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "rerank"
    assert (
        cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL)
        == "rerank-multilingual-v2.0"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {"content": {"query": query, "documents": documents}},
    )

    # Validate model response Event
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {
            "content": [
                {
                    "index": result.index,
                    "document": result.document,
                    "relevance_score": result.relevance_score,
                }
                for result in response.results
            ]
        },
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


@pytest.mark.vcr
def test_cohere_rerank_with_events_with_no_content(
    span_exporter, log_exporter, instrument_with_no_content, cohere_client
):
    query = "What is the capital of the United States?"
    documents = [
        "Carson City is the capital city of the American state of Nevada."
        + " At the  2010 United States Census, Carson City had a population of 55,274.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands"
        + " in the Pacific Ocean that are a political division controlled by the "
        + "United States. Its capital is Saipan.",
        "Charlotte Amalie is the capital and largest city of the United States "
        + "Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
        "Washington, D.C. (also known as simply Washington or D.C., and officially "
        + "as the District of Columbia) is the capital of the United States. It is a federal district. ",
        "Capital punishment (the death penalty) has existed in the United States "
        + "since before the United States was a country. As of 2017, capital "
        + "punishment is legal in 30 of the 50 states.",
        "North Dakota is a state in the United States. 672,591 people lived"
        + " in North Dakota in the year 2010. The capital and seat of government is Bismarck.",
    ]

    cohere_client.rerank(
        query=query, documents=documents, top_n=3, model="rerank-multilingual-v2.0"
    )

    spans = span_exporter.get_finished_spans()
    cohere_span = spans[0]
    assert cohere_span.name == "cohere.rerank"
    assert cohere_span.attributes.get(SpanAttributes.LLM_SYSTEM) == "Cohere"
    assert cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_TYPE) == "rerank"
    assert (
        cohere_span.attributes.get(SpanAttributes.LLM_REQUEST_MODEL)
        == "rerank-multilingual-v2.0"
    )

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate model response Event
    choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.attributes.get(EventAttributes.EVENT_NAME) == event_name
    assert (
        log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM)
        == GenAIAttributes.GenAiSystemValues.COHERE.value
    )

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
