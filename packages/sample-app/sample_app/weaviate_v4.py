# Tested with weaviate-client==4.6.3
# Weaviate instrumentation with opentelemetry-instrumentation-weaviate==0.20.0
# Code is adapted from official documentation.
# V4 documentation: https://weaviate.io/developers/weaviate/client-libraries/python
import os

import weaviate
import weaviate.classes as wvc
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow
from traceloop.sdk.instruments import Instruments


BACKEND = {
    "cohere": {
        "vectorizer": wvc.config.Configure.Vectorizer.text2vec_cohere,
        "generative": wvc.config.Configure.Generative.cohere,
    },
    "openai": {
        "vectorizer": wvc.config.Configure.Vectorizer.text2vec_openai,
        "generative": wvc.config.Configure.Generative.openai,
    },
}
COLLECTION_NAME = "Article"
RAW_QUERY = """
 {
   Get {
     Article(limit: 2) {
        author
        text
     }
   }
 }
 """


@task("create_collection")
def create_collection(client, backend):
    client.collections.create(
        name=COLLECTION_NAME,
        description="An Article class to store a text",
        vectorizer_config=backend["vectorizer"](),
        generative_config=backend["generative"](),
        properties=[
            wvc.config.Property(
                name="author",
                data_type=wvc.config.DataType.TEXT,
                description="The name of the author",
            ),
            wvc.config.Property(
                name="text",
                data_type=wvc.config.DataType.TEXT,
                description="The text content",
            ),
        ]
    )


@task("get_schema")
def get_collection(client, collection_name):
    """Get the collection to test connection"""
    return client.collections.get(collection_name)


@task("insert_data")
def insert_data(collection):
    return collection.data.insert({
        "author": "Robert",
        "text": "Once upon a time, someone wrote a book...",
    })


@task("batch_create")
def create_batch(collection):
    objs = [
        {
            "author": "Robert",
            "text": "Once upon a time, R. wrote a book...",
        },
        {
            "author": "Johnson",
            "text": "Once upon a time, J. wrote some news...",
        },
        {
            "author": "Maverick",
            "text": "Never again, M. will write a book...",
        },
        {
            "author": "Wilson",
            "text": "Lost in the island, W. did not write anything...",
        },
        {
            "author": "Ludwig",
            "text": "As king, he ruled...",
        },
    ]
    with collection.batch.dynamic() as batch:
        for obj in objs:
            batch.add_object(properties=obj)


@task("query_get")
def query_get(collection):
    return collection.query.fetch_objects(
        return_properties=["author", ]
    )


@task("query_aggregate")
def query_aggregate(collection):
    return collection.aggregate.over_all(total_count=True)


@task("query_raw")
def query_raw(client):
    return client.graphql_raw_query(RAW_QUERY)


@task("delete_collection")
def delete_collection(client):
    client.collections.delete(COLLECTION_NAME)


@task("create_schemas")
def create_schemas(client):
    client.collections.create_from_dict(
        {
            "class": "Author",
            "description": "An author that writes an article",
            "properties": [
                {
                    "name": "name",
                    "dataType": ["string"],
                    "description": "The name of the author",
                },
            ],
        },
    )
    client.collections.create_from_dict(
        {
            "class": COLLECTION_NAME,
            "description": "An Article class to store a text",
            "properties": [
                {
                    "name": "author",
                    "dataType": ["Author"],
                    "description": "The author",
                },
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "The text content",
                },
            ],
        },
    )


@task("delete_all")
def delete_all(client):
    client.collections.delete_all()


@workflow("example")
def example_workflow(client, backend):
    delete_all(client)

    create_collection(client, backend)
    print("Created collection")
    collection = get_collection(client, COLLECTION_NAME)
    print("Retrieved collection: ", collection)

    uuid = insert_data(collection)
    print("Created object of UUID: ", uuid)
    obj = collection.query.fetch_object_by_id(uuid)
    print("Retrieved obj: ", obj)

    create_batch(collection)
    result = query_get(collection)
    print("Query result:", result)
    aggregate_result = query_aggregate(collection)
    print("Aggregate result:", aggregate_result)
    raw_result = query_raw(client)
    print("Raw result: ", raw_result)

    delete_collection(client)
    print("Deleted collection")


@workflow("example2")
def example_workflow2(client):
    delete_all(client)
    create_schemas(client)


if __name__ == "__main__":
    Traceloop.init(
        app_name="weaviate_app",
        disable_batch=True,
        exporter=None if os.getenv("TRACELOOP_API_KEY") else ConsoleSpanExporter(),
        # comment below if you would like to see everything
        instruments={Instruments.WEAVIATE},
    )
    print("Traceloop initialized")

    headers = {}
    backend = None
    if (key_cohere := os.getenv("COHERE_API_KEY")) is not None:
        headers.update({"X-Cohere-Api-Key": key_cohere})
        backend = BACKEND["cohere"]
    elif (key_openai := os.getenv("OPENAI_API_KEY")) is not None:
        headers.update({"X-OpenAI-Api-Key": key_openai})
        backend = BACKEND["openai"]
    else:
        raise RuntimeError("Missing backend configuration")

    if (cluster_name := os.getenv("WEAVIATE_CLUSTER_URL")) is not None:
        client = weaviate.connect_to_wcs(
            cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
            auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
            headers=headers,
        )
    else:
        client = weaviate.connect_to_local(headers=headers)
    print("Client connected")

    example_workflow2(client)
    example_workflow(client, backend)
    delete_all(client)
    client.close()
