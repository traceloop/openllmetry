import os
import voyageai
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

Traceloop.init()

client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))


@workflow(name="embed_documents")
def embed_documents():
    """Generate embeddings for a list of documents."""
    texts = [
        "The capital of France is Paris.",
        "London is the capital of England.",
        "Berlin is the capital of Germany.",
    ]
    result = client.embed(texts=texts, model="voyage-3-lite", input_type="document")
    print(f"Generated {len(result.embeddings)} embeddings")
    print(f"Embedding dimension: {len(result.embeddings[0])}")
    print(f"Total tokens used: {result.total_tokens}")
    return result


@workflow(name="rerank_documents")
def rerank_documents():
    """Rerank documents based on relevance to a query."""
    query = "What is the capital of France?"
    documents = [
        "Paris is a beautiful city known for the Eiffel Tower.",
        "The capital of France is Paris.",
        "London has Big Ben and the Thames River.",
        "Berlin is known for its history and culture.",
    ]
    result = client.rerank(
        query=query,
        documents=documents,
        model="rerank-2-lite",
        top_k=2,
    )
    print(f"Query: {query}")
    print(f"Top {len(result.results)} results:")
    for r in result.results:
        print(f"  - Doc {r.index}: Score {r.relevance_score:.4f}")
        print(f"    {documents[r.index]}")
    print(f"Total tokens used: {result.total_tokens}")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("Voyage AI Embeddings Example")
    print("=" * 60)
    embed_documents()

    print()
    print("=" * 60)
    print("Voyage AI Reranking Example")
    print("=" * 60)
    rerank_documents()
