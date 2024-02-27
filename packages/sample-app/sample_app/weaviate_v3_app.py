# Tested with weaviate-client==3.26.0
# Code is adapted from official documentation.
# V3 documentation: https://weaviate.io/developers/weaviate/client-libraries/python_v3
# Some parts were also adapted from:
# https://towardsdatascience.com/getting-started-with-weaviate-python-client-e85d14f19e4f


import os

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task
from traceloop.sdk.decorators import workflow
import weaviate


# Init trace
Traceloop.init(
    app_name="weaviate_app", disable_batch=True
)

auth_config = weaviate.auth.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"])

client = weaviate.Client(
    url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_client_secret=auth_config,
    timeout_config=(5, 15),
    additional_headers={
        "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"],
    },
)


@task("create_schema")
def create_schema():
    client.schema.create_class(
        {
            "class": "Article",
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
def get_schema():
    return client.schema.get("Article")  # Get the schema to test connection


@task("delete_schema")
def delete_schema():
    client.schema.delete_class("Article")


@task("create_object")
def create_object():
    return client.data_object.create(
        data_object={
            "author": "Robert",
            "text": "Once upon a time, someone wrote a book...",
        },
        class_name="Article",
    )


@task("batch_create")
def create_batch():
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
            batch.add_data_object(obj, class_name="Article")


@task("query_get")
def query_get():
    return client.query.get(class_name="Article", properties=["author"]).do()


@task("query_aggregate")
def query_aggregate():
    return client.query.aggregate(class_name="Article").with_meta_count().do()


@task("query_raw")
def query_raw():
    return client.query.raw(
        """
 {
   Get {
     Article(limit: 2) {
        author
        text
     }
   }
 }
 """
    )


@task("validate")
def validate():
    return client.data_object.validate(
        data_object={
            "author": "Robert",
            "text": "Once upon a time, someone wrote a book...",
        },
        class_name="Article",
    )


@workflow("example")
def example_schema_workflow():
    create_schema()
    print("Created schema")
    result = get_schema()
    print("Retrieved schema: ", result)
    validate()
    uuid = create_object()
    print("Created object of UUID: ", uuid)
    client.data_object.exists(uuid, class_name="Article")
    obj = client.data_object.get(uuid, class_name="Article")
    print("Retrieved obj: ", obj)

    create_batch()

    result = query_get()
    print("Query result:", result)

    aggregate_result = query_aggregate()
    print("Aggregate result:", aggregate_result)

    raw_result = query_raw()
    print("Raw result: ", raw_result)
    delete_schema()
    print("Deleted schema...")


@task("create_schemas")
def create_schemas():
    client.schema.create(
        {
            "classes": [
                {
                    "class": "Article",
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
def delete_all():
    client.schema.delete_all()


@workflow("example2")
def example_schema_workflow2():
    create_schemas()
    delete_all()


delete_all()
example_schema_workflow2()
example_schema_workflow()
