import pytest
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)


@pytest.fixture
def index():
    docs = SimpleDirectoryReader("./data/paul_graham/").load_data()
    return VectorStoreIndex.from_documents(docs)


@pytest.mark.vcr
def test_chat_engine(exporter, index):
    chat_engine = index.as_chat_engine(
        similarity_top_k=3,
        chat_mode="condense_plus_context",
    )

    chat_engine.chat("What did the author do growing up?")

    spans = exporter.get_finished_spans()

    assert set(
        [
            "llama_index_chat_engine.workflow",
            "openai.embeddings",
            "get_query_embedding.task",
            "retrieve.task",
            "openai.chat",
        ]
    ) == set([span.name for span in spans])

    chat_engine_span = next(
        span for span in spans if span.name == "llama_index_chat_engine.workflow"
    )
    retriever_span = next(span for span in spans if span.name == "retrieve.task")
    query_embeddings_span = next(
        span for span in spans if span.name == "get_query_embedding.task"
    )
    chat_span = next(span for span in spans if span.name == "openai.chat")

    assert retriever_span.parent.span_id == chat_engine_span.context.span_id
    assert query_embeddings_span.parent.span_id == retriever_span.context.span_id
    assert chat_span.parent.span_id == chat_engine_span.context.span_id


@pytest.mark.vcr
def test_stream_chat_engine(exporter, index):
    chat_engine = index.as_chat_engine(
        similarity_top_k=3,
        chat_mode="condense_plus_context",
    )

    result = chat_engine.stream_chat("What did the author do growing up?")
    for _ in result.response_gen:
        pass

    spans = exporter.get_finished_spans()

    assert set(
        [
            "llama_index_chat_engine.workflow",
            "openai.embeddings",
            "get_query_embedding.task",
            "retrieve.task",
            "openai.chat",
        ]
    ) == set([span.name for span in spans])

    chat_engine_span = next(
        span for span in spans if span.name == "llama_index_chat_engine.workflow"
    )
    retriever_span = next(span for span in spans if span.name == "retrieve.task")
    query_embeddings_span = next(
        span for span in spans if span.name == "get_query_embedding.task"
    )
    chat_span = next(span for span in spans if span.name == "openai.chat")

    assert retriever_span.parent.span_id == chat_engine_span.context.span_id
    assert query_embeddings_span.parent.span_id == retriever_span.context.span_id
    assert chat_span.parent.span_id == chat_engine_span.context.span_id


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_astream_chat_engine(exporter, index):
    chat_engine = index.as_chat_engine(
        similarity_top_k=3,
        chat_mode="condense_plus_context",
    )

    result = await chat_engine.astream_chat("What did the author do growing up?")
    async for _ in result.async_response_gen():
        pass

    spans = exporter.get_finished_spans()

    assert set(
        [
            "llama_index_chat_engine.workflow",
            "openai.embeddings",
            "get_query_embedding.task",
            "retrieve.task",
            "openai.chat",
        ]
    ) == set([span.name for span in spans])

    chat_engine_span = next(
        span for span in spans if span.name == "llama_index_chat_engine.workflow"
    )
    retriever_span = next(span for span in spans if span.name == "retrieve.task")
    query_embeddings_span = next(
        span for span in spans if span.name == "get_query_embedding.task"
    )
    chat_span = next(span for span in spans if span.name == "openai.chat")

    assert retriever_span.parent.span_id == chat_engine_span.context.span_id
    assert query_embeddings_span.parent.span_id == retriever_span.context.span_id
    assert chat_span.parent.span_id == chat_engine_span.context.span_id

    spans = exporter.get_finished_spans()

    assert set(
        [
            "llama_index_chat_engine.workflow",
            "openai.embeddings",
            "get_query_embedding.task",
            "retrieve.task",
            "openai.chat",
        ]
    ) == set([span.name for span in spans])

    chat_engine_span = next(
        span for span in spans if span.name == "llama_index_chat_engine.workflow"
    )
    retriever_span = next(span for span in spans if span.name == "retrieve.task")
    query_embeddings_span = next(
        span for span in spans if span.name == "get_query_embedding.task"
    )
    chat_span = next(span for span in spans if span.name == "openai.chat")

    assert retriever_span.parent.span_id == chat_engine_span.context.span_id
    assert query_embeddings_span.parent.span_id == retriever_span.context.span_id
    assert chat_span.parent.span_id == chat_engine_span.context.span_id
