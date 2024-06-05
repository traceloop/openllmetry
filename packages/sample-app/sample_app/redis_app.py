"""_summary_
    Code was taken from https://redis.io/docs/latest/develop/get-started/vector-database/
    It requires a Redis DB running with the RedisSearch and RedisJSON modules enabled.
"""


import json
import time

import numpy as np
import pandas as pd
import redis
import requests
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

#Traceloop.init(exporter=ConsoleSpanExporter())
Traceloop.init(app_name="redis", disable_batch=True)

url = "https://raw.githubusercontent.com/bsbodden/redis_vss_getting_started/main/data/bikes.json"
response = requests.get(url)
bikes = response.json()

json.dumps(bikes[0], indent=2)

client = redis.Redis(host="localhost", port=6379, decode_responses=True)

res = client.ping()
# >>> True

pipeline = client.pipeline()
for i, bike in enumerate(bikes, start=1):
    redis_key = f"bikes:{i:03}"
    pipeline.json().set(redis_key, "$", bike)
res = pipeline.execute()
# >>> [True, True, True, True, True, True, True, True, True, True, True]

res = client.json().get("bikes:010", "$.model")
# >>> ['Summit']

keys = sorted(client.keys("bikes:*"))
# >>> ['bikes:001', 'bikes:002', ..., 'bikes:011']

descriptions = client.json().mget(keys, "$.description")
descriptions = [item for sublist in descriptions for item in sublist]
embedder = SentenceTransformer("msmarco-distilbert-base-v4")
embeddings = embedder.encode(descriptions).astype(np.float32).tolist()
VECTOR_DIMENSION = len(embeddings[0])
# >>> 768

pipeline = client.pipeline()
for key, embedding in zip(keys, embeddings):
    pipeline.json().set(key, "$.description_embeddings", embedding)
pipeline.execute()
# >>> [True, True, True, True, True, True, True, True, True, True, True]

res = client.json().get("bikes:010")
# >>>
# {
#   "model": "Summit",
#   "brand": "nHill",
#   "price": 1200,
#   "type": "Mountain Bike",
#   "specs": {
#     "material": "alloy",
#     "weight": "11.3"
#   },
#   "description": "This budget mountain bike from nHill performs well..."
#   "description_embeddings": [
#     -0.538114607334137,
#     -0.49465855956077576,
#     -0.025176964700222015,
#     ...
#   ]
# }


schema = (
    TextField("$.model", no_stem=True, as_name="model"),
    TextField("$.brand", no_stem=True, as_name="brand"),
    NumericField("$.price", as_name="price"),
    TagField("$.type", as_name="type"),
    TextField("$.description", as_name="description"),
    VectorField(
        "$.description_embeddings",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIMENSION,
            "DISTANCE_METRIC": "COSINE",
        },
        as_name="vector",
    ),
)
definition = IndexDefinition(prefix=["bikes:"], index_type=IndexType.JSON)
res = client.ft("idx:bikes_vss").create_index(
    fields=schema, definition=definition
)
# >>> 'OK'

info = client.ft("idx:bikes_vss").info()
num_docs = info["num_docs"]
indexing_failures = info["hash_indexing_failures"]
# print(f"{num_docs} documents indexed with {indexing_failures} failures")
# >>> 11 documents indexed with 0 failures

query = Query("@brand:Peaknetic")
res = client.ft("idx:bikes_vss").search(query).docs
# print(res)
# >>> [Document {'id': 'bikes:008', 'payload': None, 'brand': 'Peaknetic', 'model': 'Soothe Electric bike', 'price': '1950', 'description_embeddings': ...

query = Query("@brand:Peaknetic").return_fields("id", "brand", "model", "price")
res = client.ft("idx:bikes_vss").search(query).docs
# print(res)
# >>> [Document {'id': 'bikes:008', 'payload': None, 'brand': 'Peaknetic', 'model': 'Soothe Electric bike', 'price': '1950'}, Document {'id': 'bikes:009', 'payload': None, 'brand': 'Peaknetic', 'model': 'Secto', 'price': '430'}]

query = Query("@brand:Peaknetic @price:[0 1000]").return_fields(
    "id", "brand", "model", "price"
)
res = client.ft("idx:bikes_vss").search(query).docs
# print(res)
# >>> [Document {'id': 'bikes:009', 'payload': None, 'brand': 'Peaknetic', 'model': 'Secto', 'price': '430'}]

queries = [
    "Bike for small kids",
    "Best Mountain bikes for kids",
    "Cheap Mountain bike for kids",
    "Female specific mountain bike",
    "Road bike for beginners",
    "Commuter bike for people over 60",
    "Comfortable commuter bike",
    "Good bike for college students",
    "Mountain bike for beginners",
    "Vintage bike",
    "Comfortable city bike",
]

encoded_queries = embedder.encode(queries)
len(encoded_queries)
# >>> 11

@workflow("create_query_table")
def create_query_table(query, queries, encoded_queries, extra_params={}):
    results_list = []
    for i, encoded_query in enumerate(encoded_queries):
        result_docs = (
            client.ft("idx:bikes_vss")
            .search(
                query,
                {
                    "query_vector": np.array(
                        encoded_query, dtype=np.float32
                    ).tobytes()
                }
                | extra_params,
            )
            .docs
        )
        for doc in result_docs:
            vector_score = round(1 - float(doc.vector_score), 2)
            results_list.append(
                {
                    "query": queries[i],
                    "score": vector_score,
                    "id": doc.id,
                    "brand": doc.brand,
                    "model": doc.model,
                    "description": doc.description,
                }
            )

    # Optional: convert the table to Markdown using Pandas
    queries_table = pd.DataFrame(results_list)
    queries_table.sort_values(
        by=["query", "score"], ascending=[True, False], inplace=True
    )
    queries_table["query"] = queries_table.groupby("query")["query"].transform(
        lambda x: [x.iloc[0]] + [""] * (len(x) - 1)
    )
    queries_table["description"] = queries_table["description"].apply(
        lambda x: (x[:497] + "...") if len(x) > 500 else x
    )
    queries_table.to_markdown(index=False)



query = (
    Query("(*)=>[KNN 3 @vector $query_vector AS vector_score]")
    .sort_by("vector_score")
    .return_fields("vector_score", "id", "brand", "model", "description")
    .dialect(2)
)

create_query_table(query, queries, encoded_queries)
# >>> | Best Mountain bikes for kids     |    0.54 | bikes:003... (+ 32 more results)

hybrid_query = (
    Query("(@brand:Peaknetic)=>[KNN 3 @vector $query_vector AS vector_score]")
    .sort_by("vector_score")
    .return_fields("vector_score", "id", "brand", "model", "description")
    .dialect(2)
)
create_query_table(hybrid_query, queries, encoded_queries)
# >>> | Best Mountain bikes for kids     |    0.3  | bikes:008... (+22 more results)

range_query = (
    Query(
        "@vector:[VECTOR_RANGE $range $query_vector]=>{$YIELD_DISTANCE_AS: vector_score}"
    )
    .sort_by("vector_score")
    .return_fields("vector_score", "id", "brand", "model", "description")
    .paging(0, 4)
    .dialect(2)
)
create_query_table(
    range_query, queries[:1], encoded_queries[:1], {"range": 0.55}
)
# >>> | Bike for small kids |    0.52 | bikes:001 | Velorim    |... (+1 more result)
