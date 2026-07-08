# Adaptation of Pinecone's sampleapp to use sentence transformer
import os
from pinecone import Pinecone
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow
from sentence_transformers import SentenceTransformer

# Set this to True for first run
create_data = False
log_to_stdout = True
kwargs = {
    "app_name": "pinecone_st_app",
    "disable_batch": True,
}
if log_to_stdout:
    kwargs["exporter"] = ConsoleSpanExporter()

# Init trace
Traceloop.init(**kwargs)

# Init pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT")
)


index_name = "traceloop-dev"
print("Loading model...")
model = SentenceTransformer("intfloat/e5-small-v2")


@workflow(name="create_index")
def insert_embeddings():
    # Encode using example from https://huggingface.co/intfloat/e5-small-v2

    input_texts = [
        "query: how much protein should a female eat",
        "query: summit define",
        "passage: As a general guideline, the CDC's average "
        "requirement of protein for women ages 19 to 70 is 46 "
        "grams per day. But, as you can see from this chart, "
        "you'll need to increase that if you're expecting or "
        "training for a marathon. Check out the chart below to"
        "see how much protein you should be eating each day.",
        "passage: Definition of summit for English Language "
        "Learners. : 1  the highest point of a mountain : "
        "the top of a mountain. : 2  the highest level. : 3"
        "  a meeting or series of meetings between the leaders"
        " of two or more governments.",
    ]
    print("Creating embeddings...")
    embeddings = model.encode(input_texts, normalize_embeddings=True)

    index = pc.Index(index_name)

    data_to_insert = []
    for i in range(len(embeddings)):
        vector = embeddings[i].tolist()
        data_to_insert.append(
            {
                "id": f"id{i}",
                "values": vector,
                "metadata": {"description": f"This is for text of index {i}"},
            }
        )
    print("Inserting into index...")
    index.upsert(data_to_insert)


@task("retrieve")
def retrieve(query):
    # Encode query
    query_embeddings = model.encode(query, normalize_embeddings=True)
    vector = query_embeddings.tolist()

    # Retrieve created index
    index = pc.Index(index_name)

    # retrieve from Pinecone
    return index.query(
        vector=vector,
        top_k=3,
        include_metadata=True,
        include_values=True,
    )


@workflow(name="retrieve_query")
def run_query(query: str):
    return retrieve(query)


if create_data:
    insert_embeddings()

query = "passage: What is the definition of summit?"
run_query(query)
