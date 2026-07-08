import json
import os

import pytest
import weaviate

from opentelemetry.semconv_ai import SpanAttributes


@pytest.fixture
def client(environment):
    auth_config = weaviate.auth.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"])
    client = weaviate.Client(
        url=os.getenv("WEAVIATE_CLUSTER_URL"),
        auth_client_secret=auth_config,
        timeout_config=(5, 15),
        additional_headers={
            "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"],
        },
    )
    return client


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


def create_schemas(client: weaviate.Client):
    client.schema.create(schemas)


def create_schema(client: weaviate.Client):
    client.schema.create_class(article_schema)


def get_schema(client: weaviate.Client):
    return client.schema.get("Article")  # Get the schema to test connection


def delete_schema(client: weaviate.Client):
    client.schema.delete_class("Article")


def create_object(client: weaviate.Client):
    return client.data_object.create(
        data_object={
            "author": "Robert",
            "text": "Once upon a time, someone wrote a book...",
        },
        class_name="Article",
    )


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


def query_get(client):
    return client.query.get(class_name="Article", properties=["author"]).do()


def query_aggregate(client):
    return client.query.aggregate(class_name="Article").with_meta_count().do()


def query_raw(client):
    return client.query.raw(raw_query)


def delete_all(client: weaviate.Client):
    client.schema.delete_all()


def validate():
    return client.data_object.validate(
        data_object={
            "author": "Robert",
            "text": "Once upon a time, someone wrote a book...",
        },
        class_name="Article",
    )


@pytest.mark.vcr
def test_weaviate_delete_all(client, exporter):
    delete_all(client)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "db.weaviate.schema.delete_all")

    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_VENDOR}") == "weaviate"
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_OPERATION}") == "delete_all"


@pytest.mark.vcr
def test_weaviate_create_schemas(client, exporter):
    create_schemas(client)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "db.weaviate.schema.create")
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_VENDOR}") == "weaviate"
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_OPERATION}") == "create"
    assert (
        json.loads(span.attributes.get("db.weaviate.schema.create.schema")) == schemas
    )


@pytest.mark.vcr
def test_weaviate_create_schema(client, exporter):
    create_schema(client)

    spans = exporter.get_finished_spans()
    span = next(
        span for span in spans if span.name == "db.weaviate.schema.create_class"
    )
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_VENDOR}") == "weaviate"
    assert (
        span.attributes.get(f"{SpanAttributes.VECTOR_DB_OPERATION}") == "create_class"
    )
    assert (
        json.loads(span.attributes.get("db.weaviate.schema.create_class.schema_class"))
        == article_schema
    )


@pytest.mark.vcr
def test_weaviate_get_schema(client, exporter):
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
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_VENDOR}") == "weaviate"
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_OPERATION}") == "get"
    assert span.attributes.get("db.weaviate.schema.get.class_name") == '"Article"'


@pytest.mark.vcr
def test_weaviate_delete_schema(client, exporter):
    delete_schema(client)

    spans = exporter.get_finished_spans()
    span = next(
        span for span in spans if span.name == "db.weaviate.schema.delete_class"
    )
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_VENDOR}") == "weaviate"
    assert (
        span.attributes.get(f"{SpanAttributes.VECTOR_DB_OPERATION}") == "delete_class"
    )
    assert (
        span.attributes.get("db.weaviate.schema.delete_class.class_name") == '"Article"'
    )


@pytest.mark.vcr
def test_weaviate_create_data_object(client, exporter):
    create_object(client)

    spans = exporter.get_finished_spans()
    span = next(
        span for span in spans if span.name == "db.weaviate.data.crud_data.create"
    )
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_VENDOR}") == "weaviate"
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_OPERATION}") == "create"
    assert json.loads(
        span.attributes.get("weaviate.data.crud_data.create.data_object")
    ) == {
        "author": "Robert",
        "text": "Once upon a time, someone wrote a book...",
    }
    assert (
        span.attributes.get("weaviate.data.crud_data.create.class_name") == '"Article"'
    )


@pytest.mark.skip("Flaky test")
@pytest.mark.vcr
def test_weaviate_create_batch(client, exporter):
    create_batch(client)

    spans = exporter.get_finished_spans()
    span = next(
        span
        for span in spans
        if span.name == "db.weaviate.batch.crud_batch.add_data_object"
    )
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_VENDOR}") == "weaviate"
    assert (
        span.attributes.get(f"{SpanAttributes.VECTOR_DB_OPERATION}")
        == "add_data_object"
    )
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


@pytest.mark.vcr
def test_weaviate_query_get(client, exporter):
    query_get(client)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "db.weaviate.gql.query.get")
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_VENDOR}") == "weaviate"
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_OPERATION}") == "get"
    assert span.attributes.get("db.weaviate.query.get.class_name") == '"Article"'
    assert span.attributes.get("db.weaviate.query.get.properties") == '["author"]'

    span = next(span for span in spans if span.name == "db.weaviate.gql.filter.do")
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_VENDOR}") == "weaviate"
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_OPERATION}") == "do"


@pytest.mark.vcr
def test_weaviate_query_aggregate(client, exporter):
    query_aggregate(client)

    spans = exporter.get_finished_spans()
    span = next(
        span for span in spans if span.name == "db.weaviate.gql.query.aggregate"
    )
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_VENDOR}") == "weaviate"
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_OPERATION}") == "aggregate"
    assert span.attributes.get("db.weaviate.query.aggregate.class_name") == '"Article"'

    span = next(span for span in spans if span.name == "db.weaviate.gql.filter.do")
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_VENDOR}") == "weaviate"
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_OPERATION}") == "do"


@pytest.mark.vcr
def test_weaviate_query_raw(client, exporter):
    query_raw(client)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "db.weaviate.gql.query.raw")
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_VENDOR}") == "weaviate"
    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_OPERATION}") == "raw"
    traced_raw_query = span.attributes.get("db.weaviate.query.raw.gql_query")
    assert "Get" in traced_raw_query
    assert "Article" in traced_raw_query
    assert "author" in traced_raw_query
    assert "text" in traced_raw_query
