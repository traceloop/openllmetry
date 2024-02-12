from os import getcwd
import pytest
import json
import chromadb

chroma = chromadb.PersistentClient(path=getcwd())


@pytest.fixture
def collection():
    yield chroma.create_collection(name="Students")
    chroma.delete_collection(name="Students")


def add_documents(collection):
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

    collection.add(
        documents=[student_info, club_info, university_info],
        metadatas=[
            {"source": "student info"},
            {"source": "club info"},
            {"source": "university info"},
        ],
        ids=["id1", "id2", "id3"],
    )


def test_chroma_add(exporter, collection):
    add_documents(collection)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "chroma.add")

    assert span.attributes.get("db.system") == "chroma"
    assert span.attributes.get("db.operation") == "add"
    assert span.attributes.get("db.chroma.add.ids_count") == 3
    assert span.attributes.get("db.chroma.add.metadatas_count") == 3
    assert span.attributes.get("db.chroma.add.documents_count") == 3


def query_collection(collection):
    collection.query(
        query_texts=["What is the student name?"],
        n_results=2,
        where={"source": "student info"},
    )


def test_chroma_query(exporter, collection):
    add_documents(collection)
    query_collection(collection)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "chroma.query")

    assert span.attributes.get("db.system") == "chroma"
    assert span.attributes.get("db.operation") == "query"
    assert span.attributes.get("db.chroma.query.query_texts_count") == 1
    assert span.attributes.get("db.chroma.query.n_results") == 2
    assert span.attributes.get("db.chroma.query.where") == "{'source': 'student info'}"

    events = span.events
    assert len(events) > 0
    for i, event in enumerate(events):
        assert event.name == f"vector_db.query.result.{i}"
        ids = event.attributes.get(f"{event.name}.ids")
        distances = event.attributes.get(f"{event.name}.distances")
        documents = event.attributes.get(f"{event.name}.documents")

        # We have lists of same length as result
        assert len(ids) > 0
        assert len(ids) == len(distances)
        assert len(distances) == len(documents)

        for id_ in ids:
            assert len(id_) > 0

        for distance in distances:
            assert distance >= 0

        for document in documents:
            assert len(document) > 0
            assert isinstance(document, str)


@pytest.mark.vcr
def test_chroma_query_segment_query(exporter, collection):
    add_documents(collection)
    query_collection(collection)

    spans = exporter.get_finished_spans()
    span = next(span for span in spans if span.name == "chroma.query.segment._query")
    assert len(span.attributes.get("db.chroma.query.segment._query.collection_id")) > 0
    events = span.events
    assert len(events) > 0
    for i, event in enumerate(events):
        assert event.name == f"vector_db.query.embeddings.{i}"
        embeddings = json.loads(event.attributes.get(f"{event.name}.vector"))
        assert len(embeddings) > 100
        for number in embeddings:
            assert number >= -1 and number <= 1
