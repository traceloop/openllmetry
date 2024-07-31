import os

import pytest
from openai import OpenAI
from opentelemetry.semconv_ai import Events, Meters, SpanAttributes
from pinecone import Pinecone


@pytest.fixture
def openai_client():
    return OpenAI()


def retrieve(openai_client, index, query):
    context_limit = 3750
    res = openai_client.embeddings.create(input=[query], model="text-embedding-ada-002")

    # retrieve from Pinecone
    xq = res.data[0].embedding

    # get relevant contexts
    res = index.query(top_k=3, include_metadata=True, include_values=True, vector=xq)
    contexts = [x["metadata"]["text"] for x in res.matches]

    # build our prompt with the retrieved contexts included
    prompt_start = "Answer the question based on the context below.\n\n" + "Context:\n"
    prompt_end = f"\n\nQuestion: {query}\nAnswer:"
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= context_limit:
            prompt = prompt_start + "\n\n---\n\n".join(contexts[: i - 1]) + prompt_end
            break
        elif i == len(contexts) - 1:
            prompt = prompt_start + "\n\n---\n\n".join(contexts) + prompt_end
    return prompt, res


def complete(openai_client, prompt):
    res = openai_client.completions.create(
        model="davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    return res.choices[0].text.strip()


def run_query(openai_client, index, query: str):
    query_with_contexts, query_res = retrieve(openai_client, index, query)
    complete(openai_client, query_with_contexts)
    return query_res


@pytest.mark.skip(
    "Can't record GRPC-based tests with VCR, as it doesn't support recording GRPC requests."
)
def test_pinecone_grpc_retrieval(traces_exporter, openai_client):
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
    )
    index = pc.create_index("gen-qa-openai-fast")

    query = (
        "Which training method should I use for sentence transformers when "
        + "I only have pairs of related sentences?"
    )
    run_query(openai_client, index, query)

    spans = traces_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
        "pinecone.query",
        "openai.completion",
    ]


@pytest.mark.vcr
def test_pinecone_retrieval(traces_exporter, metrics_reader, openai_client):
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
    )
    index = pc.Index("gen-qa-openai-fast")

    query = (
        "Which training method should I use for sentence transformers when "
        + "I only have pairs of related sentences?"
    )
    query_res = run_query(openai_client, index, query)

    spans = traces_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
        "pinecone.query",
        "openai.completion",
    ]

    span = next(span for span in spans if span.name == "pinecone.query")
    assert (
        span.attributes.get("server.address")
        == "https://gen-qa-openai-fast-90c5d9e.svc.gcp-starter.pinecone.io"
    )
    assert span.attributes.get(SpanAttributes.PINECONE_QUERY_TOP_K) == 3
    assert span.attributes.get(SpanAttributes.PINECONE_USAGE_READ_UNITS) == 6
    assert span.attributes.get(SpanAttributes.PINECONE_USAGE_WRITE_UNITS) == 0
    assert span.attributes.get(SpanAttributes.PINECONE_QUERY_INCLUDE_VALUES)
    assert span.attributes.get(SpanAttributes.PINECONE_QUERY_INCLUDE_METADATA)

    events = span.events
    assert len(events) > 0

    embeddings_events = [
        event for event in span.events if Events.DB_QUERY_EMBEDDINGS.value in event.name
    ]
    for event in embeddings_events:
        assert event.name == Events.DB_QUERY_EMBEDDINGS.value
        vector = event.attributes.get(f"{event.name}.vector")
        assert len(vector) > 100
        for v in vector:
            assert v >= -1 and v <= 1

    query_result_events = [
        event for event in span.events if "db.pinecone.query.result" in event.name
    ]
    for event in query_result_events:
        assert event.name == "db.pinecone.query.result"

        id = event.attributes.get(f"{event.name}.id")
        score = event.attributes.get(f"{event.name}.score")
        metadata = event.attributes.get(f"{event.name}.metadata")
        vector = event.attributes.get(f"{event.name}.vector")
        assert len(id) > 0
        assert score > 0
        assert len(metadata) > 0
        assert len(vector) > 100
        for v in vector:
            assert v >= -1 and v <= 1

    metrics_data = metrics_reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_query_duration_metric = False
    found_scores_metric = False
    found_read_units_metric = False
    found_write_units_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:

                if metric.name == Meters.PINECONE_DB_QUERY_DURATION:
                    found_query_duration_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.sum > 0

                if metric.name == Meters.PINECONE_DB_QUERY_SCORES:
                    found_scores_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.sum == sum(
                            match.get("score") for match in query_res.get("matches")
                        )
                        assert (
                            data_point.attributes["server.address"]
                            == "https://gen-qa-openai-fast-90c5d9e.svc.gcp-starter.pinecone.io"
                        )

                if metric.name == Meters.PINECONE_DB_USAGE_READ_UNITS:
                    found_read_units_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value == 6
                        assert (
                            data_point.attributes["server.address"]
                            == "https://gen-qa-openai-fast-90c5d9e.svc.gcp-starter.pinecone.io"
                        )

                if metric.name == Meters.PINECONE_DB_USAGE_WRITE_UNITS:
                    found_write_units_metric = True
                    for data_point in metric.data.data_points:
                        assert data_point.value == 0
                        assert (
                            data_point.attributes["server.address"]
                            == "https://gen-qa-openai-fast-90c5d9e.svc.gcp-starter.pinecone.io"
                        )

    assert found_query_duration_metric is True
    assert found_scores_metric is True
    assert found_read_units_metric is True
    assert found_write_units_metric is True
