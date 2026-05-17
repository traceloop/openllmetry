import json
from os import getcwd

import chromadb
import pytest
from opentelemetry.semconv_ai import Events, SpanAttributes

chroma = chromadb.PersistentClient(path=getcwd())


@pytest.fixture
def collection():
    yield chroma.create_collection(name="Students")
    chroma.delete_collection(name="Students")


def add_documents(collection, with_metadata=False):
    student_info = """
    Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA,
    is a member of the programming and chess clubs who enjoys pizza, swimming, and hiking
    in her free time in hopes of working at a tech company after graduating from the University of Washington.
    """

    club_info = """
    The university chess club provides an outlet for students to come together and enjoy playing
    the classic strategy game of chess. Members of all skill levels are welcome, from beginners learning
    the rules to experienced tournament players. The club typically meets a few times per week to play casual games,
    participate in tournaments, analyze famous chess matches, and improve members' skills.
    """

    university_info = """
    The University of Washington, founded in 1861 in Seattle, is a public research university
    with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
    As the flagship institution of the six public universities in Washington state,
    UW encompasses over 500 buildings and 20 million square feet of space,
    including one of the largest library systems in the world."""

    if with_metadata:
        collection.add(
            documents=[student_info, club_info, university_info],
            metadatas=[
                {"source": "student info"},
                {"source": "club info"},
                {"source": "university info"},
            ],
            ids=["id1", "id2", "id3"],
        )
    else:
        collection.add(
            documents=[student_info, club_info, university_info],
            ids=["id1", "id2", "id3"],
        )


def test_chroma_add(exporter, collection):
    add_documents(collection, with_metadata=True)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "chroma.add")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "chroma"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "add"
    assert span.attributes.get(SpanAttributes.CHROMADB_ADD_IDS_COUNT) == 3
    assert span.attributes.get(SpanAttributes.CHROMADB_ADD_METADATAS_COUNT) == 3
    assert span.attributes.get(SpanAttributes.CHROMADB_ADD_DOCUMENTS_COUNT) == 3


def test_chroma_query(exporter, collection):
    add_documents(collection)
    collection.query(
        query_texts=["What is the student name?"],
        n_results=2,
    )

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "chroma.query")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "chroma"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "query"
    assert span.attributes.get(SpanAttributes.CHROMADB_QUERY_TEXTS_COUNT) == 1
    assert span.attributes.get(SpanAttributes.CHROMADB_QUERY_N_RESULTS) == 2

    events = span.events
    assert len(events) == 1
    for event in events:
        assert event.name == Events.DB_QUERY_RESULT.value
        ids_ = event.attributes.get(f"{event.name}.id")
        distance = event.attributes.get(f"{event.name}.distance")
        document = event.attributes.get(f"{event.name}.document")

        assert len(ids_) > 0
        assert isinstance(ids_, str)

        assert distance >= 0

        assert len(document) > 0
        assert isinstance(document, str)


def test_chroma_query_with_metadata(exporter, collection):
    add_documents(collection, with_metadata=True)
    collection.query(
        query_texts=["What is the student name?"],
        n_results=2,
        where={"source": "student info"},
    )

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "chroma.query")

    assert span.attributes.get(SpanAttributes.VECTOR_DB_VENDOR) == "chroma"
    assert span.attributes.get(SpanAttributes.VECTOR_DB_OPERATION) == "query"
    assert span.attributes.get(SpanAttributes.CHROMADB_QUERY_TEXTS_COUNT) == 1
    assert span.attributes.get(SpanAttributes.CHROMADB_QUERY_N_RESULTS) == 2
    assert (
        span.attributes.get(SpanAttributes.CHROMADB_QUERY_WHERE)
        == "{'source': 'student info'}"
    )

    events = span.events
    assert len(events) == 1
    for event in events:
        assert event.name == Events.DB_QUERY_RESULT.value
        ids_ = event.attributes.get(f"{event.name}.id")
        distance = event.attributes.get(f"{event.name}.distance")
        document = event.attributes.get(f"{event.name}.document")

        assert len(ids_) > 0
        assert isinstance(ids_, str)

        assert distance >= 0

        assert len(document) > 0
        assert isinstance(document, str)


def test_chroma_query_segment_query(exporter, collection):
    add_documents(collection, with_metadata=True)
    collection.query(
        query_texts=["What is the student name?"],
        n_results=2,
    )

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "chroma.query.segment._query")
    assert (
        len(
            span.attributes.get(
                SpanAttributes.CHROMADB_QUERY_SEGMENT_QUERY_COLLECTION_ID
            )
        )
        > 0
    )
    events = span.events
    assert len(events) > 0
    for event in events:
        assert event.name == Events.DB_QUERY_EMBEDDINGS.value
        embeddings = json.loads(event.attributes.get(f"{event.name}.vector"))
        assert len(embeddings) > 100
        for number in embeddings:
            assert -1 <= number <= 1
