import os
import pytest
import pinecone
from openai import OpenAI


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
    return prompt


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
    query_with_contexts = retrieve(openai_client, index, query)
    complete(openai_client, query_with_contexts)


@pytest.mark.skip("GRPC package of Pinecone is conflicting with google-cloud-aiplatform")
def test_pinecone_grpc_retrieval(exporter, openai_client):
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
    )
    index = pinecone.GRPCIndex("gen-qa-openai-fast")

    query = (
        "Which training method should I use for sentence transformers when "
        + "I only have pairs of related sentences?"
    )
    run_query(openai_client, index, query)

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
        "pinecone.query",
        "openai.completion",
    ]


@pytest.mark.vcr
def test_pinecone_retrieval(exporter, openai_client):
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
    )
    index = pinecone.Index("gen-qa-openai-fast")

    query = (
        "Which training method should I use for sentence transformers when "
        + "I only have pairs of related sentences?"
    )
    run_query(openai_client, index, query)

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.embeddings",
        "pinecone.query",
        "openai.completion",
    ]

    span = next(span for span in spans if span.name == "pinecone.query")
    assert span.attributes.get("pinecone.query.top_k") == 3
    assert span.attributes.get("pinecone.query.include_values")
    assert span.attributes.get("pinecone.query.include_metadata")

    events = span.events
    assert len(events) > 0

    embeddings_events = [event for event in span.events if "db.query.embeddings" in event.name]
    for event in embeddings_events:
        assert event.name == "db.query.embeddings"
        vector = event.attributes.get(f"{event.name}.vector")
        assert len(vector) > 100
        for v in vector:
            assert v >= -1 and v <= 1

    usage_events = [event for event in span.events if "pinecone.query.usage" in event.name]
    assert len(usage_events) == 1
    usage_event = usage_events[0]
    assert usage_event.name == "pinecone.query.usage"
    assert usage_event.attributes.get("readUnits") >= 0

    query_result_events = [event for event in span.events if "db.pinecone.query.result" in event.name]
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
