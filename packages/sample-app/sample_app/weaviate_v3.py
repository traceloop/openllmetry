# Tested with weaviate-client==3.26.0
# Code is adapted from official documentation.
# V3 documentation: https://weaviate.io/developers/weaviate/client-libraries/python/python_v3
# Some parts were also adapted from:
# https://towardsdatascience.com/getting-started-with-weaviate-python-client-e85d14f19e4f
import os

import weaviate
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task
from traceloop.sdk.decorators import workflow
from traceloop.sdk.instruments import Instruments


CLASS_NAME = "Article"
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


@task("create_schema")
def create_schema(client):
    client.schema.create_class(
        {
            "class": CLASS_NAME,
            "description": "An Article class to store a text",
            "properties": [
                {
                    "name": "author",
                    "dataType": ["string"],
                    "description": "The name of the author",
                },
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "The text content",
                },
            ],
        }
    )


@task("get_schema")
def get_schema(client):
    """Get the schema to test connection"""
    return client.schema.get(CLASS_NAME)


@task("delete_schema")
def delete_schema(client):
    client.schema.delete_class(CLASS_NAME)


@task("create_object")
def create_object(client):
    return client.data_object.create(
        data_object={
            "author": "Robert",
            "text": "Once upon a time, someone wrote a book...",
        },
        class_name=CLASS_NAME,
    )


@task("batch_create")
def create_batch(client):
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
    with client.batch as batch:
        for obj in objs:
            batch.add_data_object(obj, class_name=CLASS_NAME)


@task("query_get")
def query_get(client):
    return client.query.get(class_name=CLASS_NAME, properties=["author"]).do()


@task("query_aggregate")
def query_aggregate(client):
    return client.query.aggregate(class_name=CLASS_NAME).with_meta_count().do()


@task("query_raw")
def query_raw(client):
    return client.query.raw(RAW_QUERY)


@task("validate")
def validate(client, uuid=None):
    return client.data_object.validate(
        data_object={
            "author": "Robert",
            "text": "Once upon a time, someone wrote a book...",
        },
        uuid=uuid,
        class_name=CLASS_NAME,
    )


@task("create_schemas")
def create_schemas(client):
    client.schema.create(
        {
            "classes": [
                {
                    "class": CLASS_NAME,
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
            ]
        }
    )


@task("delete_schemas")
def delete_all(client):
    client.schema.delete_all()


@workflow("example")
def example_schema_workflow(client):
    delete_all(client)

    create_schema(client)
    print("Created schema")
    schema = get_schema(client)
    print("Retrieved schema: ", schema)
    result = validate(client)
    print(f"Object found: {result.get('valid')}")

    uuid = create_object(client)
    print("Created object of UUID: ", uuid)
    client.data_object.exists(uuid, class_name=CLASS_NAME)
    obj = client.data_object.get(uuid, class_name=CLASS_NAME)
    print("Retrieved obj: ", obj)
    result = validate(client, uuid=uuid)
    print(f"Object found: {result.get('valid')}")

    create_batch(client)
    result = query_get(client)
    print("Query result:", result)
    aggregate_result = query_aggregate(client)
    print("Aggregate result:", aggregate_result)
    raw_result = query_raw(client)
    print("Raw result: ", raw_result)

    delete_schema(client)
    print("Deleted schema")


@workflow("example2")
def example_schema_workflow2(client):
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

    additional_headers = {}
    if (key := os.getenv("COHERE_API_KEY")) is not None:
        additional_headers.update({"X-Cohere-Api-Key": key})
    elif (key := os.getenv("OPENAI_API_KEY")) is not None:
        additional_headers.update({"X-OpenAI-Api-Key": key})
    else:
        raise RuntimeError("Missing api key cohere/openai")

    if (cluster_name := os.getenv("WEAVIATE_CLUSTER_URL")) is not None:
        auth_config = weaviate.auth.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"])
        client = weaviate.Client(
            url=os.getenv("WEAVIATE_CLUSTER_URL"),
            auth_client_secret=auth_config,
            timeout_config=(5, 15),
            additional_headers=additional_headers,
        )
    else:
        client = weaviate.Client(
            url="http://localhost:8080",
            additional_headers=additional_headers,
        )
    print("Client connected")

    example_schema_workflow2(client)
    example_schema_workflow(client)
    delete_all(client)
