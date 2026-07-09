# Adaptation of Pinecone's example
# https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/gen-qa-openai.ipynb
from pinecone_datasets import load_dataset
import os
from pinecone import Pinecone
from openai import OpenAI

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

client = OpenAI()

Traceloop.init(app_name="pinecone_app")

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT")
)
index_name = "gen-qa-openai-fast"

open_ai_client = OpenAI()


@workflow(name="create_index")
def gen_index():
    if index_name not in pc.list_indexes():
        print("Loading dataset...")
        dataset = load_dataset("youtube-transcripts-text-embedding-ada-002")
        dataset.documents.drop(["metadata"], axis=1, inplace=True)
        dataset.documents.rename(columns={"blob": "metadata"}, inplace=True)

        print("Creating index...")
        pc.create_index(
            index_name,
            dimension=1536,  # dimensionality of text-embedding-ada-002
            metric="cosine",
        )
        index = pc.Index(index_name)
        for batch in dataset.iter_documents(batch_size=100):
            index.upsert(batch)


index = pc.Index(index_name)


@task("retrieve")
def retrieve(query):
    context_limit = 3750
    res = open_ai_client.embeddings.create(
        input=[query], model="text-embedding-ada-002"
    )

    # retrieve from Pinecone
    xq = res.data[0].embedding
    xq = res.data[0].embedding

    # get relevant contexts
    res = index.query(vector=xq, top_k=3, include_metadata=True, include_values=True)
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


@task("chat")
def chat(prompt):
    res = open_ai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    return res.choices[0].message.content


@workflow(name="query_with_retrieve")
def run_query(query: str):
    query_with_contexts = retrieve(query)
    print(query_with_contexts)
    print(chat(query_with_contexts))


# gen_index()
query = (
    "Which training method should I use for sentence transformers when "
    + "I only have pairs of related sentences?"
)
run_query(query)
