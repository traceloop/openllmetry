import asyncio

import chromadb
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from traceloop.sdk import Traceloop

os.environ["TOKENIZERS_PARALLELISM"] = "false"

Traceloop.init(app_name="llama_index_example")

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# define embedding function
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)


async def main():
    # Query Data
    query_engine = index.as_query_engine()
    response = await query_engine.aquery("What did the author do growing up?")

    print(response)


if __name__ == "__main__":
    asyncio.run(main())
