import os
import pytest
import pinecone
from openai import OpenAI
from traceloop.sdk.decorators import workflow, task


@pytest.fixture
def openai_client():
    return OpenAI()


@task("retrieve")
def retrieve(openai_client, index, query):
    context_limit = 3750
    res = openai_client.embeddings.create(input=[query], model="text-embedding-ada-002")

    # retrieve from Pinecone
    xq = res.data[0].embedding

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
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


@task("complete")
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


@workflow(name="query_with_retrieve")
def run_query(openai_client, index, query: str):
    query_with_contexts = retrieve(openai_client, index, query)
    complete(openai_client, query_with_contexts)


# GRPC package of Pinecone is conflicting with google-cloud-aiplatform
def disabled_test_pinecone_grpc_retrieval(exporter, openai_client):
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
        "retrieve.task",
        "openai.completion",
        "complete.task",
        "query_with_retrieve.workflow",
    ]


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
        "retrieve.task",
        "openai.completion",
        "complete.task",
        "query_with_retrieve.workflow",
    ]
