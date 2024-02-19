import pytest
import chromadb

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import OpenAIEmbedding


@pytest.mark.vcr
def test_query(exporter):
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart")

    # define embedding function
    embed_model = OpenAIEmbedding()

    # load documents
    documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

    # set up ChromaVectorStore and load in data
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, service_context=service_context
    )
    engine = index.as_query_engine()
    engine.query("What did the author do growing up?")

    spans = exporter.get_finished_spans()
    assert set(
        [
            "llama_index_retriever_query.workflow",
            "retrieve.task",
            "synthesize.task",
        ]
    ).issubset([span.name for span in spans])
