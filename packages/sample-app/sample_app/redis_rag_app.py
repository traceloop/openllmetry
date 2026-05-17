import os
import numpy as np

from redis import Redis
from redis.exceptions import ResponseError
from redis.commands.search.field import TagField, VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from openai import OpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

INDEX_NAME = "index"
DOC_PREFIX = "doc:"
EMBEDDINGS_DIM = 1536

Traceloop.init(app_name="redis_rag_app")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
rc = Redis(host="localhost", port=6379, decode_responses=True)
assert rc.ping() is True, "Cannot connect to Redis"


def load_data():
    with open("data/sherlock/firstchapter.txt") as firstchapter_file:
        data = firstchapter_file.readlines()
    return data


def prepare_embeddings(data):
    response = client.embeddings.create(input=data, model="text-embedding-3-small")
    embeddings = np.array(
        [r["embedding"] for r in response.to_dict()["data"]], dtype=np.float32
    )
    return embeddings


def upload_data_to_redis(data, embeddings):
    pipe = rc.pipeline()
    for i, embedding in enumerate(embeddings):
        pipe.hset(
            f"{DOC_PREFIX}{i}",
            mapping={
                "vector": embedding.tobytes(),
                "content": data[i],
                "tag": "openai",
            },
        )
    pipe.execute()


def create_index():
    try:
        rc.ft(INDEX_NAME).info()
        print("Index already exists")
    except ResponseError:
        schema = (
            TagField("tag"),
            TextField("content"),
            VectorField(
                "vector",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": EMBEDDINGS_DIM,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )
        definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH)
        rc.ft(INDEX_NAME).create_index(fields=schema, definition=definition)


def query_redis(query_embeddings):
    query = (
        Query("(@tag:{ openai })=>[KNN 2 @vector $vec as score]")
        .sort_by("score")
        .return_fields("content", "tag", "score")
        .paging(0, 2)
        .dialect(2)
    )
    query_params = {"vec": query_embeddings.tobytes()}
    response = rc.ft(INDEX_NAME).search(query, query_params).docs
    relevant_content = [doc["content"] for doc in response]
    return relevant_content


def get_response_from_openai(query, relevant_content):
    messages = [
        {
            "role": "system",
            "content": """You are a helpfull assistant that answers questions about book: A Study in Scarlet.
Narrator in the book is doctor Watson. User will provide you a relevant content of the book and a question""",
        },
        {
            "role": "user",
            "content": f"Based on that content: {relevant_content} - Answer my question: {query}",
        },
    ]
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
        max_tokens=100,
    )

    for part in stream:
        print(part.choices[0].delta.content or "", end="")
    print()


@workflow("redis_rag_app")
def main():
    data = load_data()
    embeddings = prepare_embeddings(data)
    upload_data_to_redis(data, embeddings)
    create_index()
    query = "Who introduced Sherlock Holmes to doctor Watson?"
    print(query)
    query_emb = prepare_embeddings(query)[0]
    relevant_content = query_redis(query_emb)
    get_response_from_openai(query, relevant_content)


main()
