# pip install pandas, sentence_transformers
import pathlib

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
import pandas as pd
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow


Traceloop.init(
    app_name="chroma_sentence_transformer_app",
    disable_batch=True,
    exporter=ConsoleSpanExporter(),
)

embedding_function = SentenceTransformerEmbeddingFunction()

chroma_client = chromadb.Client()

data_root_dir = pathlib.Path(__file__).parent.parent / "data"

claim_df = pd.read_json(
    (data_root_dir / "scifact/scifact_claims.jsonl").resolve(),
    lines=True,
).head(10)

corpus_df = pd.read_json(
    (data_root_dir / "scifact/scifact_corpus.jsonl").resolve(),
    lines=True,
).head(10)

scifact_corpus_collection = chroma_client.create_collection(
    name="scifact_corpus", embedding_function=embedding_function
)

batch_size = 100

for i in range(0, len(corpus_df), batch_size):
    batch_df = corpus_df[i:i + batch_size]
    scifact_corpus_collection.add(
        ids=batch_df["doc_id"]
        .apply(lambda x: str(x))
        .tolist(),  # Chroma takes string IDs.
        documents=(
            batch_df["title"] + ". " + batch_df["abstract"].apply(lambda x: " ".join(x))
        ).to_list(),  # We concatenate the title and abstract.
        metadatas=[
            {"structured": structured}
            for structured in batch_df["structured"].to_list()
        ],  # We also store the metadata, though we don't use it in this example.
    )


def build_prompt_with_context(claim, context):
    return [
        {
            "role": "system",
            "content": "I will ask you to assess whether a particular scientific claim, based on evidence provided. "
            + "Output only the text 'True' if the claim is true, 'False' if the claim is false, or 'NEE' if there's "
            + "not enough evidence.",
        },
        {
            "role": "user",
            "content": f""""
The evidence is the following:

{' '.join(context)}

Assess the following claim on the basis of the evidence. Output only the text 'True' if the claim is true,
'False' if the claim is false, or 'NEE' if there's not enough evidence. Do not output any other text.

Claim:
{claim}

Assessment:
""",
        },
    ]


@workflow("assess_claims")
def assess_claims(claims):
    scifact_corpus_collection.query(
        query_texts=claims, include=["documents", "distances"], n_results=3
    )


samples = claim_df.sample(2)
assess_claims(samples["claim"].tolist())
