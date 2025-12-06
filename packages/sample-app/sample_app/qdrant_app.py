# pip install qdrant-client sentence-transformers
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import numpy as np

from traceloop.sdk import Traceloop
Traceloop.init(app_name="qdrant_instrumentation_app")

client = QdrantClient(":memory:")

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [
    "Qdrant is an open-source vector database.",
    "It supports fast similarity search over embeddings.",
    "You can combine metadata filters with vector search."
]
embeddings = model.encode(texts, show_progress_bar=False)
vector_size = embeddings.shape[1]

collection_name = "demo_collection"

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
)

points = [
    models.PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": texts[i]})
    for i in range(len(texts))
]
client.upsert(collection_name=collection_name, points=points)

query = "What is Qdrant?"
q_vec = model.encode(query).tolist()

resp = client.query_points(
    collection_name=collection_name,
    query=q_vec,
    with_payload=True,
    limit=3,   
)

for p in resp.points:
    print(f"id={p.id} score={p.score:.4f} payload={p.payload}")