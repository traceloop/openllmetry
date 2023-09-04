# Adaptation of Pinecone's example
# https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/gen-qa-openai.ipynb

from dotenv import load_dotenv

load_dotenv()

from pinecone_datasets import load_dataset
import os
import pinecone
import openai


pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT")
)
index_name = "gen-qa-openai-fast"
if index_name not in pinecone.list_indexes():
    print("Loading dataset...")
    dataset = load_dataset("youtube-transcripts-text-embedding-ada-002")
    dataset.documents.drop(["metadata"], axis=1, inplace=True)
    dataset.documents.rename(columns={"blob": "metadata"}, inplace=True)

    print("Creating index...")
    pinecone.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002
        metric="cosine",
    )
    index = pinecone.GRPCIndex(index_name)
    for batch in dataset.iter_documents(batch_size=100):
        index.upsert(batch)
index = pinecone.GRPCIndex(index_name)


def retrieve(query):
    context_limit = 3750
    res = openai.Embedding.create(input=[query], engine="text-embedding-ada-002")

    # retrieve from Pinecone
    xq = res["data"][0]["embedding"]

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    contexts = [x["metadata"]["text"] for x in res["matches"]]

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


query = (
    "Which training method should I use for sentence transformers when "
    + "I only have pairs of related sentences?"
)
query_with_contexts = retrieve(query)
print(query_with_contexts)
print(complete(query_with_contexts))
