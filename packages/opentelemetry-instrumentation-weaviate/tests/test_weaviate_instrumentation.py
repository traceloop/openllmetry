import json
import os

import pytest
from traceloop.sdk.decorators import task
import weaviate


@pytest.fixture(scope="session")
def client():
    auth_config = weaviate.auth.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"])
    client = weaviate.Client(
        url=os.getenv("WEAVIATE_CLUSTER_URL"),
        auth_client_secret=auth_config,
        timeout_config=(5, 15),
        additional_headers={
            "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"],
        },
    )
    yield client


schemas = {
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

article_schema = {
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

raw_query = """
 {
   Get {
     Article(limit: 2) {
        author
        text
     }
   }
 }
 """


@task("create_schemas")
def create_schemas(client: weaviate.Client):
    client.schema.create(schemas)


@task("create_schema")
def create_schema(client: weaviate.Client):
    client.schema.create_class(article_schema)


@task("get_schema")
def get_schema(client: weaviate.Client):
    return client.schema.get("Article")  # Get the schema to test connection


@task("delete_schema")
def delete_schema(client: weaviate.Client):
    client.schema.delete_class("Article")


@task("create_object")
def create_object(client: weaviate.Client):
    return client.data_object.create(
        data_object={
            "author": "Robert",
            "text": "Once upon a time, someone wrote a book...",
        },
        class_name="Article",
    )


@task("batch_create")
def create_batch(client: weaviate.Client):
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
def query_get(client):
    return client.query.get(class_name="Article", properties=["author"]).do()


@task("query_aggregate")
def query_aggregate(client):
    return client.query.aggregate(class_name="Article").with_meta_count().do()


@task("query_raw")
def query_raw(client):
    return client.query.raw(raw_query)


@task("delete_schemas")
def delete_all(client: weaviate.Client):
    client.schema.delete_all()


@task("validate")
def validate():
    return client.data_object.validate(
        data_object={
            "author": "Robert",
            "text": "Once upon a time, someone wrote a book...",
        },
        class_name="Article",
    )


def test_weaviate_delete_all(client, exporter):
    delete_all(client)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "db.weaviate.schema.delete_all")

    assert span.attributes.get("db.system") == "weaviate"
    assert span.attributes.get("db.operation") == "delete_all"


def test_weaviate_create_schemas(client, exporter):
    delete_all(client)
    create_schemas(client)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "db.weaviate.schema.create")
    assert span.attributes.get("db.system") == "weaviate"
    assert span.attributes.get("db.operation") == "create"
    assert (
        json.loads(span.attributes.get("db.weaviate.schema.create.schema")) == schemas
    )


def test_weaviate_create_schema(client, exporter):
    delete_all(client)
    create_schema(client)

    spans = exporter.get_finished_spans()
    span = next(
        span for span in spans if span.name == "db.weaviate.schema.create_class"
    )
    assert span.attributes.get("db.system") == "weaviate"
    assert span.attributes.get("db.operation") == "create_class"
    assert (
        json.loads(span.attributes.get("db.weaviate.schema.create_class.schema_class"))
        == article_schema
    )


def test_weaviate_get_schema(client, exporter):
    delete_all(client)
    create_schema(client)
    get_schema(client)

    spans = exporter.get_finished_spans()

    # Apparently the tracing capture additional calls to 'schema.get', potentially from 'delete_all' call
    # Therefore, we have to filter by the one where we passed the class name.
    span = next(
        span
        for span in spans
        if span.name == "db.weaviate.schema.get"
        and span.attributes.get("db.weaviate.schema.get.class_name")
    )
    print(span.attributes)
    assert span.attributes.get("db.system") == "weaviate"
    assert span.attributes.get("db.operation") == "get"
    assert span.attributes.get("db.weaviate.schema.get.class_name") == '"Article"'


def test_weaviate_delete_schema(client, exporter):
    delete_all(client)
    create_schema(client)
    delete_schema(client)

    spans = exporter.get_finished_spans()
    span = next(
        span for span in spans if span.name == "db.weaviate.schema.delete_class"
    )
    assert span.attributes.get("db.system") == "weaviate"
    assert span.attributes.get("db.operation") == "delete_class"
    assert (
        span.attributes.get("db.weaviate.schema.delete_class.class_name") == '"Article"'
    )


def test_weaviate_create_data_object(client, exporter):
    delete_all(client)
    create_schema(client)
    create_object(client)

    spans = exporter.get_finished_spans()
    span = next(
        span for span in spans if span.name == "db.weaviate.data.crud_data.create"
    )
    assert span.attributes.get("db.system") == "weaviate"
    assert span.attributes.get("db.operation") == "create"
    assert json.loads(
        span.attributes.get("weaviate.data.crud_data.create.data_object")
    ) == {
        "author": "Robert",
        "text": "Once upon a time, someone wrote a book...",
    }
    assert (
        span.attributes.get("weaviate.data.crud_data.create.class_name") == '"Article"'
    )


def test_weaviate_create_batch(client, exporter):
    delete_all(client)
    create_schema(client)
    create_batch(client)

    spans = exporter.get_finished_spans()
    span = next(
        span
        for span in spans
        if span.name == "db.weaviate.batch.crud_batch.add_data_object"
    )
    assert span.attributes.get("db.system") == "weaviate"
    assert span.attributes.get("db.operation") == "add_data_object"
    data_object = json.loads(
        span.attributes.get("db.weaviate.batch.add_data_object.data_object")
    )
    assert data_object["author"] in [
        "Robert",
        "Johnson",
        "Maverick",
        "Wilson",
        "Ludwig",
    ]
    assert "..." in data_object["text"]
    assert (
        span.attributes.get("db.weaviate.batch.add_data_object.class_name")
        == '"Article"'
    )


def test_weaviate_query_get(client, exporter):
    delete_all(client)
    create_schema(client)
    create_batch(client)
    query_get(client)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "db.weaviate.gql.query.get")
    assert span.attributes.get("db.system") == "weaviate"
    assert span.attributes.get("db.operation") == "get"
    assert span.attributes.get("db.weaviate.query.get.class_name") == '"Article"'
    assert span.attributes.get("db.weaviate.query.get.properties") == '["author"]'

    span = next(
        span for span in spans if span.name == "db.weaviate.gql.query.filter.do"
    )
    assert span.attributes.get("db.system") == "weaviate"
    assert span.attributes.get("db.operation") == "do"


def test_weaviate_query_aggregate(client, exporter):
    delete_all(client)
    create_schema(client)
    create_batch(client)
    query_aggregate(client)

    spans = exporter.get_finished_spans()
    span = next(
        span for span in spans if span.name == "db.weaviate.gql.query.aggregate"
    )
    assert span.attributes.get("db.system") == "weaviate"
    assert span.attributes.get("db.operation") == "aggregate"
    assert span.attributes.get("db.weaviate.query.aggregate.class_name") == '"Article"'

    span = next(
        span for span in spans if span.name == "db.weaviate.gql.query.filter.do"
    )
    assert span.attributes.get("db.system") == "weaviate"
    assert span.attributes.get("db.operation") == "do"


def test_weaviate_query_raw(client, exporter):
    delete_all(client)
    create_schema(client)
    create_batch(client)
    query_raw(client)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "db.weaviate.gql.query.raw")
    assert span.attributes.get("db.system") == "weaviate"
    assert span.attributes.get("db.operation") == "raw"
    traced_raw_query = span.attributes.get("db.weaviate.query.raw.gql_query")
    assert "Get" in traced_raw_query
    assert "Article" in traced_raw_query
    assert "author" in traced_raw_query
    assert "text" in traced_raw_query
