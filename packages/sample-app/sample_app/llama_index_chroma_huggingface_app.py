import asyncio
import os

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.vector_stores.chroma import ChromaVectorStore

from traceloop.sdk import Traceloop


async def main():
    # Query Data
    query_engine = index.as_query_engine()
    response = await query_engine.aquery("What did the author do growing up?")
    print(response)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    Traceloop.init(app_name="llama_index_example")

    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart")

    # set llm and embed model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = HuggingFaceInferenceAPI(
        model_name="HuggingFaceH4/zephyr-7b-alpha", token=os.environ["HUGGING_FACE_API_KEY"]
    )

    # load documents
    documents = SimpleDirectoryReader(r"../data/paul_graham/").load_data()

    # set up ChromaVectorStore and load in data
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context,
    )

    asyncio.run(main())
