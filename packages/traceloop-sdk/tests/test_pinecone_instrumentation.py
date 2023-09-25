import os
import pinecone
import openai
from traceloop.sdk.decorators import workflow, task


def test_pinecone_retrieval(exporter):
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
    )
    index = pinecone.GRPCIndex("gen-qa-openai-fast")

    @task("retrieve")
    def retrieve(query):
        context_limit = 3750
        res = openai.Embedding.create(input=[query], engine="text-embedding-ada-002")

        # retrieve from Pinecone
        xq = res["data"][0]["embedding"]

        # get relevant contexts
        res = index.query(xq, top_k=3, include_metadata=True)
        contexts = [x["metadata"]["text"] for x in res["matches"]]

        # build our prompt with the retrieved contexts included
        prompt_start = (
            "Answer the question based on the context below.\n\n" + "Context:\n"
        )
        prompt_end = f"\n\nQuestion: {query}\nAnswer:"
        # append contexts until hitting limit
        for i in range(1, len(contexts)):
            if len("\n\n---\n\n".join(contexts[:i])) >= context_limit:
                prompt = (
                    prompt_start + "\n\n---\n\n".join(contexts[: i - 1]) + prompt_end
                )
                break
            elif i == len(contexts) - 1:
                prompt = prompt_start + "\n\n---\n\n".join(contexts) + prompt_end
        return prompt

    @task("complete")
    def complete(prompt):
        res = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=400,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        return res["choices"][0]["text"].strip()

    @workflow(name="query_with_retrieve")
    def run_query(query: str):
        query_with_contexts = retrieve(query)
        complete(query_with_contexts)

    query = (
        "Which training method should I use for sentence transformers when "
        + "I only have pairs of related sentences?"
    )
    run_query(query)

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "pinecone.query",
        "retrieve.task",
        "openai.completion",
        "complete.task",
        "query_with_retrieve.workflow",
    ]
