import json

import pytest
import weaviate
import weaviate.classes as wvc

from opentelemetry.semconv_ai import SpanAttributes


ARTICLE_SCHEMA = {
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


@pytest.fixture(name="client")
def fixture_client():
    with weaviate.connect_to_local(skip_init_checks=True) as client:
        client.collections.delete_all()
        yield client
        client.collections.delete_all()


def create_collection(wclient: weaviate.WeaviateClient) -> None:
    wclient.collections.create(
        name="Article",
        description="An Article class to store a text",
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
        ],
    )


def create_collection_from_dict(wclient: weaviate.WeaviateClient) -> None:
    wclient.collections.create_from_dict(ARTICLE_SCHEMA)


def get_collection(
    wclient: weaviate.WeaviateClient,
) -> weaviate.collections.collection.Collection:
    return wclient.collections.get("Article")


def delete_collection(wclient: weaviate.WeaviateClient) -> None:
    wclient.collections.delete("Article")


def insert_data(wclient: weaviate.WeaviateClient) -> str:
    collection = wclient.collections.get("Article")
    result = collection.data.insert(
        {
            "author": "Robert",
            "text": "Once upon a time, someone wrote a book...",
        },
        uuid="0fa769e6-2f64-4717-97e3-f0297cf84138",
    )
    return str(result)  # uuid is not JSON serializable, so convert


def create_batch(wclient: weaviate.WeaviateClient) -> None:
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
    collection = wclient.collections.get("Article")
    with collection.batch.dynamic() as batch:
        for obj in objs:
            batch.add_object(properties=obj)


def query_fetch_object_by_id(wclient: weaviate.WeaviateClient, uuid_value: str):
    collection = wclient.collections.get("Article")
    return collection.query.fetch_object_by_id(uuid_value, return_properties=None)


def query_fetch_objects(wclient: weaviate.WeaviateClient):
    collection = wclient.collections.get("Article")
    return collection.query.fetch_objects(return_properties=["author"])


def query_aggregate(wclient: weaviate.WeaviateClient):
    collection = wclient.collections.get("Article")
    return collection.aggregate.over_all(total_count=True)


def query_raw(wclient):
    return wclient.graphql_raw_query(RAW_QUERY)


def delete_all(wclient: weaviate.WeaviateClient):
    wclient.collections.delete_all()


@pytest.mark.vcr
def test_weaviate_delete_all(client, exporter):
    create_collection(client)
    delete_all(client)

    spans = exporter.get_finished_spans()
    span = next(
        span for span in spans if span.name == "db.weaviate.collections.delete_all"
    )

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "weaviate"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "delete_all"


@pytest.mark.vcr
def test_weaviate_create_collection(client, exporter):
    create_collection(client)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "db.weaviate.collections.create")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "weaviate"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "create"
    assert span.attributes.get("db.weaviate.collections.create.name") == '"Article"'


@pytest.mark.vcr
def test_weaviate_create_collection_from_dict(client, exporter):
    create_collection_from_dict(client)

    spans = exporter.get_finished_spans()
    span = next(
        span
        for span in spans
        if span.name == "db.weaviate.collections.create_from_dict"
    )

    assert span.attributes.get(f"{SpanAttributes.VECTOR_DB_VENDOR}") == "weaviate"
    assert (
        span.attributes.get(f"{SpanAttributes.VECTOR_DB_OPERATION}")
        == "create_from_dict"
    )
    assert (
        json.loads(
            span.attributes.get("db.weaviate.collections.create_from_dict.config")
        )
        == ARTICLE_SCHEMA
    )


@pytest.mark.vcr
def test_weaviate_get_collection(client, exporter):
    create_collection(client)
    _ = get_collection(client)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "db.weaviate.collections.get")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "weaviate"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "get"
    assert span.attributes.get("db.weaviate.collections.get.name") == '"Article"'


@pytest.mark.vcr
def test_weaviate_delete_collection(client, exporter):
    create_collection(client)
    delete_collection(client)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "db.weaviate.collections.delete")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "weaviate"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "delete"
    assert span.attributes.get("db.weaviate.collections.delete.name") == '"Article"'


@pytest.mark.vcr
def test_weaviate_insert_data(client, exporter):
    create_collection(client)
    _ = insert_data(client)

    spans = exporter.get_finished_spans()
    span = next(
        span for span in spans if span.name == "db.weaviate.collections.data.insert"
    )

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "weaviate"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "insert"
    assert json.loads(
        span.attributes.get("weaviate.collections.data.insert.properties")
    ) == {
        "author": "Robert",
        "text": "Once upon a time, someone wrote a book...",
    }


@pytest.mark.vcr
def test_weaviate_create_batch(client, exporter):
    create_collection(client)
    create_batch(client)

    spans = exporter.get_finished_spans()
    span = next(
        span
        for span in spans
        if span.name == "db.weaviate.collections.batch.add_object"
    )

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "weaviate"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "add_object"
    data_object = json.loads(
        span.attributes.get("db.weaviate.collections.batch.add_object.properties")
    )
    assert data_object["author"] in [
        "Robert",
        "Johnson",
        "Maverick",
        "Wilson",
        "Ludwig",
    ]
    assert "..." in data_object["text"]


@pytest.mark.with_grpc
def test_weaviate_query_fetch_object_by_id(client, exporter):
    create_collection(client)
    uuid_value = insert_data(client)
    data = query_fetch_object_by_id(client, uuid_value)

    spans = exporter.get_finished_spans()
    span = next(
        span
        for span in spans
        if span.name == "db.weaviate.collections.query.fetch_object_by_id"
    )

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "weaviate"
    assert (
        span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "fetch_object_by_id"
    )
    assert (
        span.attributes.get("db.weaviate.collections.query.fetch_object_by_id.uuid")
        == f'"{uuid_value}"'
    )
    assert data.properties.get("author") == "Robert"


@pytest.mark.with_grpc
def test_weaviate_query_fetch_objects(client, exporter):
    create_collection(client)
    create_batch(client)
    result = query_fetch_objects(client)

    spans = exporter.get_finished_spans()
    span = next(
        span
        for span in spans
        if span.name == "db.weaviate.collections.query.fetch_objects"
    )

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "weaviate"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "fetch_objects"
    assert (
        span.attributes.get(
            "db.weaviate.collections.query.fetch_objects.return_properties"
        )
        == '["author"]'
    )
    assert len(result.objects) != 0


@pytest.mark.vcr
def test_weaviate_query_aggregate(client, exporter):
    create_collection(client)
    create_batch(client)
    result = query_aggregate(client)

    spans = exporter.get_finished_spans()
    span_aggregate = next(
        span for span in spans if span.name == "db.weaviate.gql.aggregate.do"
    )
    span_filter = next(
        span for span in spans if span.name == "db.weaviate.gql.filter.do"
    )

    assert span_aggregate.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "weaviate"
    assert span_aggregate.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "do"
    assert result.total_count == 5
    assert span_filter.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "weaviate"
    assert span_filter.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "do"


@pytest.mark.vcr
def test_weaviate_query_raw(client, exporter):
    create_collection(client)
    create_batch(client)
    result = query_raw(client)

    spans = exporter.get_finished_spans()
    span = next(
        span for span in spans if span.name == "db.weaviate.client.graphql_raw_query"
    )
    traced_raw_query = span.attributes.get(
        "db.weaviate.client.graphql_raw_query.gql_query"
    )

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "weaviate"
    assert (
        span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "graphql_raw_query"
    )
    assert "Get" in traced_raw_query
    assert "Article" in traced_raw_query
    assert "author" in traced_raw_query
    assert "text" in traced_raw_query
    assert len(result.get["Article"]) == 2
