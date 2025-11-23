import json

import pytest
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import ChatCohere
from opentelemetry.sdk._logs import LogData
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes

# source: wikipedia
INPUT_TEXT = """
    Today, all ridges and faces of the Matterhorn have been ascended in all seasons,
    and mountain guides take a large number of people up the northeast Hörnli route
    each summer. In total, up to 150 climbers attempt the Matterhorn each day during
    summer. By modern standards, the climb is fairly difficult (AD Difficulty rating),
    but not hard for skilled mountaineers according to French climbing grades. There
    are fixed ropes on parts of the route to help. Still, it should be remembered that
    several climbers may die on the mountain each year.
    The usual pattern of ascent is to take the Schwarzsee cable car up from Zermatt,
    hike up to the Hörnli Hut elev. 3,260 m (10,700 ft), a large stone building at the
    base of the main ridge, and spend the night. The next day, climbers rise at 3:30 am
    so as to reach the summit and descend before the regular afternoon clouds and storms
    come in. The Solvay Hut located on the ridge at 4,003 m (13,133 ft) can be used only
    in a case of emergency.
    Other popular routes on the mountain include the Italian (Lion) ridge (AD+ Difficulty
    rating) and the Zmutt ridge (D Difficulty rating). The four faces, as well as the
    Furggen ridge, constitute the most challenging routes to the summit. The north face
    is amongst the six most difficult faces of the Alps, as well as ‘The Trilogy’, the
    three hardest of the six, along with the north faces of the Eiger and the Grandes
    Jorasses (TD+ Difficulty rating).
"""


@pytest.mark.vcr
def test_sequential_chain(instrument_legacy, span_exporter, log_exporter):
    small_docs = CharacterTextSplitter().create_documents(
        texts=[
            INPUT_TEXT,
        ]
    )
    llm = ChatCohere(model="command", temperature=0.75)
    chain = load_summarize_chain(llm, chain_type="stuff").with_config(
        run_name="stuff_chain"
    )
    chain.invoke(small_docs)

    spans = span_exporter.get_finished_spans()

    assert [
        "ChatCohere.chat",
        "LLMChain.task",
        "stuff_chain.workflow",
    ] == [span.name for span in spans]

    stuff_span = next(span for span in spans if span.name == "stuff_chain.workflow")

    data = json.loads(stuff_span.attributes[SpanAttributes.TRACELOOP_ENTITY_INPUT])
    assert data["inputs"].keys() == {"input_documents"}
    assert data["kwargs"]["name"] == "stuff_chain"
    data = json.loads(stuff_span.attributes[SpanAttributes.TRACELOOP_ENTITY_OUTPUT])
    assert data["outputs"].keys() == {"output_text"}

    logs = log_exporter.get_finished_logs()
    assert (
        len(logs) == 0
    ), "Assert that it doesn't emit logs when use_legacy_attributes is True"


@pytest.mark.vcr
def test_sequential_chain_with_events_with_content(
    instrument_with_content, span_exporter, log_exporter
):
    small_docs = CharacterTextSplitter().create_documents(
        texts=[
            INPUT_TEXT,
        ]
    )
    llm = ChatCohere(model="command", temperature=0.75)
    chain = load_summarize_chain(llm, chain_type="stuff").with_config(
        run_name="stuff_chain"
    )
    response = chain.invoke(small_docs)

    spans = span_exporter.get_finished_spans()

    assert [
        "ChatCohere.chat",
        "LLMChain.task",
        "stuff_chain.workflow",
    ] == [span.name for span in spans]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(
        logs[0],
        "gen_ai.user.message",
        {
            "content": 'Write a concise summary of the following:\n\n\n"{}"\n\n\nCONCISE SUMMARY:'.format(
                response["input_documents"][0].page_content
            ),
        },
    )

    # Validate AI choice Event
    _choice_event = {
        "index": 0,
        "finish_reason": "unknown",
        "message": {"content": response["output_text"]},
    }
    assert_message_in_logs(logs[1], "gen_ai.choice", _choice_event)


@pytest.mark.vcr
def test_sequential_chain_with_events_with_no_content(
    instrument_with_no_content, span_exporter, log_exporter
):
    small_docs = CharacterTextSplitter().create_documents(
        texts=[
            INPUT_TEXT,
        ]
    )
    llm = ChatCohere(model="command", temperature=0.75)
    chain = load_summarize_chain(llm, chain_type="stuff").with_config(
        run_name="stuff_chain"
    )
    chain.invoke(small_docs)

    spans = span_exporter.get_finished_spans()

    assert [
        "ChatCohere.chat",
        "LLMChain.task",
        "stuff_chain.workflow",
    ] == [span.name for span in spans]

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 2

    # Validate user message Event
    assert_message_in_logs(logs[0], "gen_ai.user.message", {})

    # Validate AI choice Event
    choice_event = {"index": 0, "finish_reason": "unknown", "message": {}}
    assert_message_in_logs(logs[1], "gen_ai.choice", choice_event)


def assert_message_in_logs(log: LogData, event_name: str, expected_content: dict):
    assert log.log_record.event_name == event_name
    assert log.log_record.attributes.get(GenAIAttributes.GEN_AI_SYSTEM) == "langchain"

    if not expected_content:
        assert not log.log_record.body
    else:
        assert log.log_record.body
        assert dict(log.log_record.body) == expected_content
