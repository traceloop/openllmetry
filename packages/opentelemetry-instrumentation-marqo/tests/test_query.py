import marqo
import pytest
from opentelemetry.semconv_ai import Events, SpanAttributes

mq = marqo.Client(url="http://localhost:8882")


@pytest.fixture
@pytest.mark.vcr
def collection():
    yield mq.create_index("TestIndex", model="hf/e5-base-v2")
    mq.index("TestIndex").delete()


@pytest.mark.vcr
def add_documents(collection):
    mq.index("TestIndex").add_documents(
        documents=[
            {
                "Title": "The Travels of Marco Polo",
                "Description": "A 13th-century travelogue describing Polo's travels",
            },
            {
                "Title": "Extravehicular Mobility Unit (EMU)",
                "Description": "The EMU is a spacesuit that provides environmental protection, "
                "mobility, life support, and communications for astronauts",
                "_id": "article_591",
            },
        ],
        tensor_fields=["Description"],
    )


@pytest.mark.vcr
def test_marqo_add_documents(exporter, collection):
    add_documents(collection)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "marqo.add_documents")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "marqo"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "add_documents"
    assert span.attributes.get("db.chroma.add.documents_count") == 2


@pytest.mark.vcr
def test_marqo_search(exporter, collection):
    add_documents(collection)
    mq.index("TestIndex").search(
        q="What is the best outfit to wear on the moon?",
    )

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "marqo.search")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "marqo"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "search"

    assert (
        span.attributes.get("db.marqo.search.query")
        == "What is the best outfit to wear on the moon?"
    )
    assert span.attributes.get("db.marqo.search.processing_time") >= 0

    events = span.events
    assert len(events) == 2
    for event in events:
        assert event.name == Events.DB_QUERY_RESULT.value
        id = event.attributes.get("_id")
        score = event.attributes.get("_score")
        title = event.attributes.get("Title")

        assert len(id) > 0
        assert isinstance(id, str)

        assert score >= 0

        assert len(title) > 0
        assert isinstance(title, str)


@pytest.mark.vcr
def test_marqo_delete_documents(exporter, collection):
    add_documents(collection)
    mq.index("TestIndex").delete_documents(ids=["article_591"])

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "marqo.delete_documents")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "marqo"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "delete_documents"

    assert span.attributes.get("db.milvus.delete.ids_count") == 1
    assert span.attributes.get("db.marqo.delete_documents.status") == "succeeded"
